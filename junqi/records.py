"""棋谱 .jqr 读写与复盘。

一局 = 28 字节头 + 120 字节布阵 + 每步 2 字节（from, to）。碰子结果不存，复盘时由引擎重算。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .engine import Game, RuleConfig, make_config

_HEAD = struct.Struct("<4s8BHHQbBH")
assert _HEAD.size == 28


@dataclass
class GameRecord:
    cfg: RuleConfig
    seed: int
    winner: int  # 胜方队伍 0/1，和棋 -1
    end_reason: int
    setups: np.ndarray  # int8[120]
    moves: np.ndarray  # uint8[n,2]

    @property
    def plies(self) -> int:
        return len(self.moves)

    def to_bytes(self) -> bytes:
        c = self.cfg
        head = _HEAD.pack(b"JQR1", 1, c.mode, c.visibility, c.arc_mode, c.center_middle_rail, c.remove_eliminated,
                          c.reveal_flag_on_marshal_death, c.camp_corner_straight, c.max_plies, c.max_plies_without_combat, self.seed,
                          self.winner, self.end_reason, len(self.moves))
        return head + self.setups.astype(np.int8).tobytes() + self.moves.astype(np.uint8).tobytes()


def parse(buf: bytes) -> list[GameRecord]:
    out: list[GameRecord] = []
    off, n = 0, len(buf)
    mv = memoryview(buf)
    while off < n:
        (magic, ver, mode, vis, arc, cmr, rem, rev, corner, maxp, maxnc, seed, winner, reason, nplies) = _HEAD.unpack_from(buf, off)
        if magic != b"JQR1" or ver != 1:
            raise ValueError(f"bad record at offset {off}")
        off += _HEAD.size
        setups = np.frombuffer(mv[off:off + 120], dtype=np.int8).copy()
        off += 120
        moves = np.frombuffer(mv[off:off + 2 * nplies], dtype=np.uint8).reshape(nplies, 2).copy()
        off += 2 * nplies
        cfg = make_config(mode=mode, visibility=vis, arc_mode=arc, center_middle_rail=cmr, remove_eliminated=rem,
                          reveal_flag_on_marshal_death=rev, max_plies=maxp, max_plies_without_combat=maxnc,
                          camp_corner_straight=corner)
        out.append(GameRecord(cfg, seed, winner, reason, setups, moves))
    return out


def read(path: str | Path) -> list[GameRecord]:
    return parse(Path(path).read_bytes())


def iter_files(paths) -> Iterator[GameRecord]:
    for p in paths:
        yield from read(p)


def write(path: str | Path, records: list[GameRecord], append: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab" if append else "wb") as f:
        for r in records:
            f.write(r.to_bytes())


def replay(rec: GameRecord, game: Game | None = None):
    """逐步复盘：yield (game, ply, seat, from, to)，yield 时尚未走这一步。"""
    g = game if game is not None else Game(rec.cfg)
    g.reset(rec.seed, rec.setups)
    for ply, (frm, to) in enumerate(rec.moves):
        yield g, ply, g.current_seat, int(frm), int(to)
        if g.step(int(frm), int(to)) < 0:
            raise ValueError(f"illegal move in record at ply {ply}: {frm}->{to}")
