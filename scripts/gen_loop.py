#!/usr/bin/env python3
"""分代训练循环：一代接一代训练，按停止规则和预算结束，中途做一次“针对性检查”。

  nohup python3 scripts/gen_loop.py --after rl_four_v6 --next 7 --budget-h 50 > runs/gen_loop.out 2>&1 &

规则：
  - 每代从上一代继续，结束时对上一代下 --final-games 局（summary.json 的 vs_bc）。
  - 连续两代得分低于 --min-score 就停；预算不够再训一代也停。
  - 花到 --exploit-at-h 小时时，专门训练一个只打当前最强模型的模型（不和它自己学，不限制偏离），
    得分越高说明漏洞越大。这个模型不进入后面的代。
控制：runs/gen_loop.STOP 存在就在下一代开始前停；runs/gen_loop.extra 的内容会追加到下一代的命令行。
记录写到 runs/gen_loop.jsonl，结束时写一行 "finished"。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("runs")


def next_action(elapsed_h: float, scores: list[float], exploited: bool, a) -> str:
    """下一步：stop_score / stop_budget / exploit / train。"""
    if len(scores) >= 2 and all(s < a.min_score for s in scores[-2:]):
        return "stop_score"
    if elapsed_h + a.gen_hours > a.budget_h:
        return "stop_budget"
    if not exploited and elapsed_h >= a.exploit_at_h:
        return "exploit"
    return "train"


def train_cmd(gen: int, chain: list[str], a) -> list[str]:
    prev = chain[-1]
    pool = [str(RUNS / r / "model.pt") for r in chain[-3:-1]]
    return [sys.executable, "-m", "junqi.train_rl", "--init", str(RUNS / prev / "model.pt"), "--pool-add", *pool,
            "--eval-opp", str(RUNS / prev / "model.pt"), "--out", str(RUNS / f"rl_four_v{gen}"), "--iters", str(a.iters),
            "--amp", "bf16", "--lr", "1.5e-5", "--pool-size", "8", "--p-self", "0.5", "--p-pool", "0.3", "--p-rule", "0.2",
            "--p-exploit", "0.5", "--draw-reward", "-0.3", "--elim-reward", "0.2", "--reveal-penalty", "0.03",
            "--shape-anneal", str(a.iters * 3 // 4), "--eval-every", "20", "--eval-games", "400", "--style-games", "200",
            "--final-games", str(a.final_games), "--setup-params", a.setup_params, "--setup-mix", "0.3", "--seed", str(gen)]


def exploit_cmd(target: str, a) -> list[str]:
    path = str(RUNS / target / "model.pt")
    return [sys.executable, "-m", "junqi.train_rl", "--init", path, "--eval-opp", path, "--out", str(RUNS / f"rl_exploit_{target}"),
            "--iters", str(a.exploit_iters), "--amp", "bf16", "--lr", "3e-5", "--kl-ref", "0", "--pool-size", "1",
            "--p-self", "0", "--p-pool", "1", "--p-rule", "0", "--eval-every", "20", "--eval-games", "400",
            "--final-games", str(a.final_games), "--seed", "1"]


def run(cmd: list[str], out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.out", "a", encoding="utf-8") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--after", required=True, help="等这个 run 训练完再开始")
    ap.add_argument("--earlier", nargs="*", default=[], help="更早的 run（从旧到新），放进前两代的对手池")
    ap.add_argument("--next", type=int, required=True, help="第一代的版本号")
    ap.add_argument("--t0", type=float, default=None, help="预算起算时间（epoch 秒），默认现在")
    ap.add_argument("--budget-h", type=float, default=50)
    ap.add_argument("--gen-hours", type=float, default=4.5, help="一代大约多少小时（判断预算够不够）")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--min-score", type=float, default=0.53)
    ap.add_argument("--final-games", type=int, default=1000)
    ap.add_argument("--exploit-at-h", type=float, default=25)
    ap.add_argument("--exploit-iters", type=int, default=100)
    ap.add_argument("--setup-params", default="runs/init/prev_heuristic.json")
    a = ap.parse_args()
    if not Path(a.setup_params).exists():  # 默认是上一版启发式布阵
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from junqi.setup_search import base_params
        Path(a.setup_params).parent.mkdir(parents=True, exist_ok=True)
        Path(a.setup_params).write_text(json.dumps({"params": base_params().tolist()}), encoding="utf-8")

    t0 = a.t0 or time.time()
    log = open(RUNS / "gen_loop.jsonl", "a", encoding="utf-8")

    def note(**kw):
        log.write(json.dumps({"t": time.time(), "elapsed_h": round((time.time() - t0) / 3600, 2), **kw}, ensure_ascii=False) + "\n")
        log.flush()

    def score_of(r: str) -> float:
        return json.loads((RUNS / r / "summary.json").read_text(encoding="utf-8"))["vs_bc"]["score"]

    while not (RUNS / a.after / "summary.json").exists():
        alive = subprocess.run(["pgrep", "-f", f"[-]-out runs/{a.after} "], capture_output=True).returncode == 0
        if not alive:
            time.sleep(120)  # summary.json 可能刚好在写
            if not (RUNS / a.after / "summary.json").exists():
                note(event="after_run_died", run=a.after)
                note(event="finished", reason="after_run_died")
                return
        time.sleep(60)

    chain = [*a.earlier, a.after]
    scores = [score_of(a.after)]
    note(event="start", run=a.after, score=scores[-1])
    gen, exploited = a.next, False
    while True:
        if (RUNS / "gen_loop.STOP").exists():
            note(event="finished", reason="stop_file")
            return
        action = next_action((time.time() - t0) / 3600, scores, exploited, a)
        if action.startswith("stop"):
            note(event="finished", reason=action, scores=scores, best=chain[-1])
            return
        if action == "exploit":
            exploited = True
            out = RUNS / f"rl_exploit_{chain[-1]}"
            rc = run(exploit_cmd(chain[-1], a), out)
            note(event="exploit", target=chain[-1], rc=rc, score=score_of(out.name) if rc == 0 else None)
            if rc != 0:
                note(event="finished", reason="exploit_failed")
                return
            continue
        extra = RUNS / "gen_loop.extra"
        cmd = train_cmd(gen, chain, a) + (shlex.split(extra.read_text(encoding="utf-8")) if extra.exists() else [])
        name = f"rl_four_v{gen}"
        note(event="train", run=name, cmd=" ".join(cmd[2:]))
        rc = run(cmd, RUNS / name)
        if rc != 0:
            note(event="finished", reason="train_failed", run=name, rc=rc)
            return
        scores.append(score_of(name))
        chain.append(name)
        note(event="done", run=name, score=scores[-1])
        gen += 1


if __name__ == "__main__":
    main()
