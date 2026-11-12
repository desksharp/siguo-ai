"""用神经网络模型自对弈产数据：四家都是同一个模型，按温度采样，输出与 gen_data 相同的 .jqr 分片和 manifest.json。

强化学习后的模型比规则 AI 强得多，用它的对局做行为克隆，可以把大模型直接带到它的水平。

示例：
  python -m junqi.gen_nn --model runs/rl_four_v1/model.pt --games 200000 --out data/v3_nn/rl1_s0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import records
from .engine import load_rules
from .model import load
from .train_rl import Rollout


class _Recorder(Rollout):
    """对局结束时顺带记下结束原因（Episode 里没有这一项）。"""

    def __init__(self, *args):
        self.reasons: list[int] = []
        super().__init__(*args)

    def _finish(self, i: int) -> None:
        self.reasons.append(self.games[i].end_reason)
        super()._finish(i)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rules", default="siguo_standard")
    ap.add_argument("--games", type=int, default=100000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=1.0, help="走子温度（1 = 强化学习训练时的策略）")
    ap.add_argument("--slots", type=int, default=512)
    ap.add_argument("--shard-games", type=int, default=20000)
    ap.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_rules(a.rules)
    model = load(a.model, a.device).requires_grad_(False)
    c = model.cfg
    if (c.arc_mode, c.center_middle_rail, c.camp_corner_straight) != (
            cfg.arc_mode, cfg.center_middle_rail, cfg.camp_corner_straight):
        raise SystemExit("模型的棋盘规则与 --rules 不一致")
    ra = SimpleNamespace(slots=a.slots, seed=a.seed, amp=a.amp, tau=a.tau, draw_reward=0.0, lam=0.95, max_lag=0,
                         p_self=1.0, p_pool=0.0, p_rule=0.0)
    a.out.mkdir(parents=True, exist_ok=True)
    roll = _Recorder(cfg, model, [], ra, a.device)

    manifest = {"rules": a.rules, "config": cfg.to_dict(), "mix": f"nn:{a.model.parent.name}@{a.tau}",
                "model": str(a.model), "tau": a.tau, "seed": a.seed, "shards": [], "created": time.time()}
    done, shard, t0 = 0, 0, time.time()
    buf: list[records.GameRecord] = []
    while done < a.games:
        roll.tick()
        for ep, reason in zip(roll.done, roll.reasons):
            buf.append(records.GameRecord(cfg, ep.seed, ep.winner, reason, np.asarray(ep.setups, np.int8),
                                          np.asarray(ep.moves, np.uint8).reshape(-1, 2)))
        roll.done, roll.reasons = [], []
        if len(buf) >= a.shard_games or (buf and done + len(buf) >= a.games):
            buf = buf[:a.games - done]
            path = a.out / f"shard_{shard:04d}.jqr"
            records.write(path, buf)
            st = {"team0": sum(r.winner == 0 for r in buf), "team1": sum(r.winner == 1 for r in buf),
                  "draws": sum(r.winner < 0 for r in buf), "plies": int(sum(r.plies for r in buf))}
            manifest["shards"].append({"file": path.name, **st})
            done += len(buf)
            shard += 1
            buf = []
            # 每个分片都更新 manifest，中途停掉也能直接拿已产出的分片训练
            manifest["games"] = done
            manifest["plies"] = sum(s["plies"] for s in manifest["shards"])
            (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {path.name}: {st}  ({done}/{a.games}, {time.time() - t0:.0f}s)", flush=True)
    print(f"done: {done} games, {manifest['plies']} plies → {a.out}")


if __name__ == "__main__":
    main()
