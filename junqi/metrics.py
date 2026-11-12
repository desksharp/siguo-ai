"""信念质量评估：硬约束信念（掩码 × 配额先验）对敌方暗子身份的预测准确率，按对局进度分段。

同一套指标之后用来评估神经信念头（传入 predict_fn 即可）。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .engine import QUOTA, TYPE_NAMES, Game
from .records import GameRecord

BITS = np.arange(12, dtype=np.uint16)


def mask_prior_probs(masks: np.ndarray) -> np.ndarray:
    """[N] uint16 掩码 → [N,12] 概率（掩码内按兵种配额加权）。"""
    m = ((masks[:, None] >> BITS) & 1).astype(np.float64) * QUOTA[None, :]
    s = m.sum(1, keepdims=True)
    return np.where(s > 0, m / np.maximum(s, 1e-12), 1.0 / 12)


def belief_report(recs: list[GameRecord], fractions=(0.3, 0.6, 0.9), predict_fn=None, max_games: int = 300) -> dict:
    """predict_fn(game, observer) -> [129,12] 概率（绝对节点编号）；缺省用硬约束信念。"""
    acc = defaultdict(list)
    nll = defaultdict(list)
    known = defaultdict(list)
    key_prob = defaultdict(lambda: defaultdict(list))
    flag_hit = defaultdict(list)
    for rec in recs[:max_games]:
        targets = sorted({max(1, int(f * rec.plies)) for f in fractions})
        g = Game(rec.cfg).reset(rec.seed, rec.setups)
        for ply, (frm, to) in enumerate(rec.moves):
            if ply in targets:
                f = fractions[targets.index(ply)] if len(targets) == len(fractions) else round(ply / rec.plies, 1)
                types, owners = g.board()
                for o in range(4):
                    if not g.seat_alive(o):
                        continue
                    team = g.team_of(o)
                    enemy = np.flatnonzero((owners >= 0) & np.array([owners[n] >= 0 and g.team_of(int(owners[n])) != team
                                                                      for n in range(129)]))
                    if len(enemy) == 0:
                        continue
                    masks = g.belief(o)[enemy]
                    probs = mask_prior_probs(masks) if predict_fn is None else predict_fn(g, o)[enemy]
                    truth = types[enemy].astype(np.int64)
                    pt = probs[np.arange(len(enemy)), truth]
                    acc[f].extend((probs.argmax(1) == truth).tolist())
                    nll[f].extend((-np.log(np.maximum(pt, 1e-9))).tolist())
                    known[f].extend((np.bitwise_count(masks) == 1).tolist() if hasattr(np, "bitwise_count")
                                    else [bin(int(m)).count("1") == 1 for m in masks])
                    for t in (0, 2, 11):  # 军旗 炸弹 司令
                        sel = truth == t
                        if sel.any():
                            key_prob[f][TYPE_NAMES[t]].extend(pt[sel].tolist())
                    # 军旗定位：每个敌方座位，p(军旗) 最大的子是否就是军旗
                    for s in set(owners[enemy].tolist()):
                        idx = owners[enemy] == s
                        if (truth[idx] == 0).any():
                            flag_hit[f].append(bool(truth[idx][probs[idx, 0].argmax()] == 0))
            g.step(int(frm), int(to))
    out = {}
    for f in sorted(acc):
        out[f] = {
            "pieces": len(acc[f]),
            "top1": float(np.mean(acc[f])),
            "nll": float(np.mean(nll[f])),
            "exactly_known": float(np.mean(known[f])),
            "flag_located": float(np.mean(flag_hit[f])) if flag_hit[f] else float("nan"),
            **{f"p({k})": float(np.mean(v)) for k, v in key_prob[f].items()},
        }
    return out


def format_report(rep: dict, title: str = "") -> str:
    cols = ["pieces", "top1", "nll", "exactly_known", "flag_located", "p(司令)", "p(炸弹)", "p(军旗)"]
    lines = [f"### {title}" if title else "", "| 进度 | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for f, r in rep.items():
        vals = [f"{r.get(c, float('nan')):.3f}" if c != "pieces" else str(r[c]) for c in cols]
        lines.append(f"| {int(f * 100)}% | " + " | ".join(vals) + " |")
    return "\n".join(x for x in lines if x)
