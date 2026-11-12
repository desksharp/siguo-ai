"""M2 预训练：在自对弈棋谱上做行为克隆 + 敌子身份预测 + 终局价值预测。

  python3 -m junqi.train_bc --data data/v2_rule/duel --out runs/bc_duel_v1 --steps 20000

训练中定期在留出分片上评估，结束后对随机 AI / 规则 AI 打对战并输出信念准确率报告。
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import records
from .arena import play_games, random_agent, rule_agent
from .agents.neural import NeuralAgent
from .dataset import batches, shard_files
from .engine import load_rules
from .metrics import belief_report, format_report
from .model import JunqiNet, NetConfig, losses, save


def to_torch(b: dict, device: str) -> dict:
    out = {}
    for k, v in b.items():
        t = torch.from_numpy(v)
        if k == "btype":
            t = t.long()
        out[k] = t.to(device, non_blocking=True)
    return out


def background(gen, depth: int = 8):
    """后台线程预取批次（C++ 复盘在 ctypes 调用时释放 GIL）。"""
    q: queue.Queue = queue.Queue(depth)

    def run():
        for item in gen:
            q.put(item)
        q.put(None)

    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def evaluate(model, files, device, n_batches: int = 20, batch_size: int = 256) -> dict:
    model.eval()
    acc: dict[str, list] = {}
    with torch.no_grad():
        for i, b in enumerate(batches(files, batch_size, buffer=4 * batch_size, keep_prob=0.3, seed=999, loop=False)):
            if i >= n_batches:
                break
            _, st = losses(model, to_torch(b, device))
            for k, v in st.items():
                acc.setdefault(k, []).append(v)
    model.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rules", default="duel", help="对战评估用的规则")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--keep-prob", type=float, default=0.25)
    ap.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--match-games", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    files = shard_files(a.data)
    if len(files) < 2:
        raise SystemExit("need at least 2 shards (last one is held out for evaluation)")
    train_files, held = files[:-1], files[-1:]
    rules = load_rules(a.rules)
    model = JunqiNet(NetConfig(d_model=a.d_model, layers=a.layers, heads=a.heads, arc_mode=rules.arc_mode,
                               center_middle_rail=rules.center_middle_rail,
                               camp_corner_straight=rules.camp_corner_straight)).to(a.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M  device: {a.device}  train shards: {len(train_files)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(a.amp)
    # torch < 2.3 只有 torch.cuda.amp.GradScaler（租用的 4090 服务器是 2.2）
    scaler_cls = getattr(torch.amp, "GradScaler", None) or torch.cuda.amp.GradScaler
    scaler = scaler_cls(enabled=a.amp == "fp16")
    log = open(a.out / "log.jsonl", "a", encoding="utf-8")
    board = {k: getattr(rules, k) for k in ("arc_mode", "center_middle_rail", "camp_corner_straight")}
    (a.out / "args.json").write_text(json.dumps({**vars(a), "out": str(a.out), "params": n_params, "board": board}, indent=2))

    gen = background(batches(train_files, a.batch, keep_prob=a.keep_prob, seed=a.seed))
    t0 = time.time()
    run: dict[str, list] = {}
    for step in range(1, a.steps + 1):
        b = to_torch(next(gen), a.device)
        with torch.autocast(device_type=a.device.split(":")[0], dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, st = losses(model, b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        for k, v in st.items():
            run.setdefault(k, []).append(v)
        if step % 100 == 0:
            rec = {"step": step, "lr": sched.get_last_lr()[0], "samples_per_s": step * a.batch / (time.time() - t0),
                   **{k: float(np.mean(v)) for k, v in run.items()}}
            run = {}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step:6d}  loss {rec['loss']:.3f}  pi_top1 {rec['policy_top1']:.3f}  "
                  f"belief_top1 {rec['belief_top1']:.3f}  value_acc {rec['value_acc']:.3f}  "
                  f"{rec['samples_per_s']:.0f} samp/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, held, a.device)
            log.write(json.dumps({"step": step, "eval": ev}) + "\n")
            log.flush()
            print(f"  [held-out] {ev}", flush=True)
            save(model, a.out / "model.pt", {"step": step})

    # ---- 结束评估：对战 + 信念报告
    cfg = load_rules(a.rules)
    agent = NeuralAgent(model, a.device, temperature=0.0, name="bc")
    summary: dict = {"params": n_params, "steps": a.steps}
    for opp in (random_agent(), rule_agent(0.1), rule_agent(0.0)):
        r = play_games(cfg, agent, opp, a.match_games, seed=123, batch=128)
        print(r, flush=True)
        summary[f"vs_{opp.name}"] = {"score": r.score, "elo": r.elo, "w": r.wins, "l": r.losses, "d": r.draws}
    recs = records.read(held[0])[:200]
    base = belief_report(recs)
    neural = belief_report(recs, predict_fn=agent.belief_probs)
    rep = format_report(base, "硬约束信念（基线）") + "\n\n" + format_report(neural, "神经信念头")
    print(rep)
    summary["belief_baseline"] = base
    summary["belief_neural"] = neural
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    (a.out / "belief_report.md").write_text(rep, encoding="utf-8")


if __name__ == "__main__":
    main()
