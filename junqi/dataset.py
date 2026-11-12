"""训练样本：复盘 .jqr 棋谱，为每一步的行动方生成（观测, 合法着, 着法, 敌子真身份, 终局结果）。

样本全部以行动方为 0 号座位的相对编号：
  node   float32[129, F]     节点特征
  glob   float32[G]          全局特征
  legal  uint8[129*129]      合法着掩码（from*129+to）
  action int64               实际着法 from*129+to
  btype  int8[129]           信念头标签：对行动方隐藏身份的敌/友子的真实兵种，其余 -1
  value  int64               行动方队伍的终局结果 0=胜 1=和 2=负

特权信息（btype）只作为训练标签，从不进入网络输入（泄漏测试见 tests/test_python.py）。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import numpy as np

from . import records
from .engine import Game, board_info

N = 129


def shard_files(paths) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        out.extend(sorted(p.rglob("*.jqr")) if p.is_dir() else [p])
    return out


def game_samples(rec: records.GameRecord, game: Game, keep_prob: float = 1.0, rng: random.Random | None = None):
    """复盘一局，yield 每步样本（dict of numpy）。"""
    B = board_info(rec.cfg.arc_mode, rec.cfg.center_middle_rail, rec.cfg.camp_corner_straight)
    rng = rng or random.Random(0)
    game.reset(rec.seed, rec.setups)
    for frm, to in rec.moves:
        frm, to = int(frm), int(to)
        seat = game.current_seat
        if keep_prob >= 1.0 or rng.random() < keep_prob:
            node, glob, legal = game.encode(seat)
            rel = B.rel[seat]
            true_rel = game.true_types(seat)
            # 标签：行动方无法确定身份的子（信念掩码不是单点）
            known = node[:, 31] > 0.5
            occupied = node[:, 8] < 0.5
            btype = np.where(occupied & ~known, true_rel, -1).astype(np.int8)
            if rec.winner < 0:
                value = 1
            else:
                value = 0 if game.team_of(seat) == rec.winner else 2
            yield {
                "node": node,
                "glob": glob,
                "legal": legal.reshape(-1),
                "action": np.int64(rel[frm] * N + rel[to]),
                "btype": btype,
                "value": np.int64(value),
            }
        if game.step(frm, to) < 0:
            raise ValueError("illegal move while replaying record")


def iter_samples(files, keep_prob: float = 0.25, seed: int = 0, loop: bool = True,
                 worker: int = 0, num_workers: int = 1) -> Iterator[dict]:
    """在分片间随机穿插遍历；keep_prob 对每局步数做子采样，降低同局样本相关性。"""
    rng = random.Random(seed * 1000 + worker)
    files = [f for i, f in enumerate(files) if i % num_workers == worker] or list(files)
    games: dict = {}
    while True:
        order = list(files)
        rng.shuffle(order)
        for f in order:
            recs = records.read(f)
            rng.shuffle(recs)
            for rec in recs:
                key = (rec.cfg.mode, rec.cfg.arc_mode, rec.cfg.center_middle_rail, rec.cfg.camp_corner_straight)
                if key not in games:
                    games[key] = Game(rec.cfg)
                yield from game_samples(rec, games[key], keep_prob, rng)
        if not loop:
            return


def collate(samples: list[dict]) -> dict:
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}


def batches(files, batch_size: int, buffer: int = 20000, **kw) -> Iterator[dict]:
    """带洗牌缓冲区的 numpy 批次生成器（无 torch 依赖）。"""
    rng = random.Random(kw.get("seed", 0) + 17)
    buf: list[dict] = []
    for s in iter_samples(files, **kw):
        buf.append(s)
        if len(buf) >= buffer:
            idx = rng.sample(range(len(buf)), batch_size)
            yield collate([buf[i] for i in idx])
            for i in sorted(idx, reverse=True):
                buf[i] = buf[-1]
                buf.pop()
    while len(buf) >= batch_size:
        yield collate(buf[:batch_size])
        buf = buf[batch_size:]
