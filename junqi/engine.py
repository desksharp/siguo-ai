"""C++ 引擎的 ctypes 封装（无需 pybind11 / Python 头文件）。"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TYPE_NAMES = ["军旗", "地雷", "炸弹", "工兵", "排长", "连长", "营长", "团长", "旅长", "师长", "军长", "司令"]
OUTCOME_NAMES = {0: "move", 1: "kill", 2: "killed", 3: "both", 4: "capture_flag", -1: "illegal"}
END_REASONS = {0: "none", 1: "team_eliminated", 2: "max_plies", 3: "no_combat"}
QUOTA = np.array([1, 3, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1], dtype=np.int64)
AGENT_RANDOM, AGENT_RULE, AGENT_EXPLOIT = 0, 1, 2
SETUP_RANDOM, SETUP_HEURISTIC, SETUP_LEGACY, SETUP_CUSTOM = 0, 1, 2, 3

_CFG_FIELDS = [
    "mode",
    "visibility",
    "arc_mode",
    "center_middle_rail",
    "remove_eliminated",
    "reveal_flag_on_marshal_death",
    "max_plies",
    "max_plies_without_combat",
    "camp_corner_straight",
]


class RuleConfig(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in _CFG_FIELDS]

    def to_dict(self) -> dict:
        return {n: int(getattr(self, n)) for n in _CFG_FIELDS}


class SelfPlaySpec(ctypes.Structure):
    _fields_ = [
        ("cfg", RuleConfig),
        ("agent_kind", ctypes.c_int32 * 4),
        ("agent_param", ctypes.c_float * 4),
        ("setup_kind", ctypes.c_int32 * 4),
    ]


_ENUMS = {
    "mode": {"four": 0, "duel": 1},
    "visibility": {"dark": 0, "team": 1, "all": 2},
    "arc_mode": {"none": 0, "front_rows": 1, "side_rails": 2, "both": 3},
}


def make_config(**overrides) -> RuleConfig:
    base = {
        "mode": 0,
        "visibility": 0,
        "arc_mode": 2,
        "center_middle_rail": 1,
        "remove_eliminated": 1,
        "reveal_flag_on_marshal_death": 1,
        "max_plies": 2000,
        "max_plies_without_combat": 400,
        "camp_corner_straight": 0,
    }
    for k, v in overrides.items():
        if k not in base:
            raise KeyError(f"unknown rule field: {k}")
        if isinstance(v, str):
            v = _ENUMS[k][v]
        base[k] = int(v)
    return RuleConfig(**base)


def load_rules(name_or_path: str | Path = "siguo_standard", **overrides) -> RuleConfig:
    """读取 configs/rules/<name>.json（或任意路径）为 RuleConfig。"""
    p = Path(name_or_path)
    if not p.suffix:
        p = ROOT / "configs" / "rules" / f"{name_or_path}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    fields = {k: v for k, v in data.items() if k in _CFG_FIELDS}
    fields.update(overrides)
    return make_config(**fields)


@lru_cache(maxsize=1)
def lib() -> ctypes.CDLL:
    name = "junqi.dll" if os.name == "nt" else "libjunqi.so"
    path = os.environ.get("JUNQI_LIB") or str(Path(__file__).resolve().parent / "_lib" / name)
    L = ctypes.CDLL(path)
    P = ctypes.c_void_p
    i8p, i16p, i32p = (ctypes.POINTER(t) for t in (ctypes.c_int8, ctypes.c_int16, ctypes.c_int32))
    u8p, u16p, u64p = (ctypes.POINTER(t) for t in (ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint64))
    i64p, f32p = ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float)
    sig = {
        "jq_version": ([], ctypes.c_int),
        "jq_dims": ([i32p], None),
        "jq_board_info": ([ctypes.c_int32] * 3 + [i16p, u8p, i8p, i16p, u8p], None),
        "jq_board_straight": ([ctypes.c_int32] * 3 + [ctypes.c_int] * 3, ctypes.c_int),
        "jq_game_new": ([ctypes.POINTER(RuleConfig)], P),
        "jq_game_free": ([P], None),
        "jq_game_reset": ([P, ctypes.c_uint64, i8p, i32p], None),
        "jq_game_load_position": ([P, i8p, i8p, ctypes.c_int, ctypes.c_uint64], ctypes.c_int),
        "jq_game_current_seat": ([P], ctypes.c_int),
        "jq_game_over": ([P], ctypes.c_int),
        "jq_game_winner": ([P], ctypes.c_int),
        "jq_game_end_reason": ([P], ctypes.c_int),
        "jq_game_ply": ([P], ctypes.c_int),
        "jq_game_team_of": ([P, ctypes.c_int], ctypes.c_int),
        "jq_game_seat_alive": ([P, ctypes.c_int], ctypes.c_int),
        "jq_game_flag_revealed": ([P, ctypes.c_int], ctypes.c_int),
        "jq_game_legal_moves": ([P, i16p, ctypes.c_int], ctypes.c_int),
        "jq_game_step": ([P, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        "jq_game_encode": ([P, ctypes.c_int, f32p, f32p, u8p], None),
        "jq_game_true_types": ([P, ctypes.c_int, i8p], None),
        "jq_game_board": ([P, i8p, i8p], None),
        "jq_game_belief": ([P, ctypes.c_int, u16p], None),
        "jq_game_setups": ([P, i8p], None),
        "jq_game_history": ([P, i16p, i8p, ctypes.c_int], ctypes.c_int),
        "jq_rng_seed": ([u64p, ctypes.c_uint64], None),
        "jq_agent_act": ([P, ctypes.c_int, ctypes.c_float, u64p, i16p], ctypes.c_int),
        "jq_valid_setup": ([i8p], ctypes.c_int),
        "jq_generate_setup": ([ctypes.c_int, u64p, i8p], None),
        "jq_set_setup_params": ([ctypes.c_int, f32p], None),
        "jq_resolve": ([ctypes.c_int, ctypes.c_int], ctypes.c_int),
        "jq_selfplay": (
            [ctypes.POINTER(SelfPlaySpec), ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, i64p],
            ctypes.c_int64,
        ),
    }
    for fname, (args, res) in sig.items():
        fn = getattr(L, fname)
        fn.argtypes = args
        fn.restype = res
    return L


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


@lru_cache(maxsize=1)
def dims() -> dict:
    out = np.zeros(6, dtype=np.int32)
    lib().jq_dims(_ptr(out, ctypes.c_int32))
    return dict(nodes=int(out[0]), node_features=int(out[1]), global_features=int(out[2]), types=int(out[3]))


@dataclass(frozen=True)
class BoardInfo:
    xy: np.ndarray  # [129,2] 17×17 网格坐标
    is_rail: np.ndarray
    is_camp: np.ndarray  # 行营
    is_base: np.ndarray  # 大本营
    seat: np.ndarray  # 营区归属，九宫 -1
    row: np.ndarray
    col: np.ndarray
    rel: np.ndarray  # [4,129] 绝对 → 相对编号
    edge_kind: np.ndarray  # [129,129] 0 无 1 公路 2 铁路
    node_at: dict

    @property
    def abs_of(self) -> np.ndarray:
        inv = np.zeros_like(self.rel)
        for o in range(4):
            inv[o, self.rel[o]] = np.arange(self.rel.shape[1])
        return inv


@lru_cache(maxsize=16)
def board_info(arc_mode: int = 2, center_middle_rail: int = 1, camp_corner_straight: int = 0) -> BoardInfo:
    n = 129
    xy = np.zeros((n, 2), np.int16)
    flags = np.zeros((n, 3), np.uint8)
    src = np.zeros((n, 3), np.int8)
    rel = np.zeros((4, n), np.int16)
    ek = np.zeros((n, n), np.uint8)
    lib().jq_board_info(
        arc_mode,
        center_middle_rail,
        camp_corner_straight,
        _ptr(xy, ctypes.c_int16),
        _ptr(flags, ctypes.c_uint8),
        _ptr(src, ctypes.c_int8),
        _ptr(rel, ctypes.c_int16),
        _ptr(ek, ctypes.c_uint8),
    )
    node_at = {(int(x), int(y)): i for i, (x, y) in enumerate(xy)}
    return BoardInfo(xy, flags[:, 0].astype(bool), flags[:, 1].astype(bool), flags[:, 2].astype(bool),
                     src[:, 0], src[:, 1], src[:, 2], rel.astype(np.int64), ek, node_at)


def is_straight(a: int, b: int, c: int, arc_mode: int = 2, center_middle_rail: int = 1,
                camp_corner_straight: int = 0) -> bool:
    return bool(lib().jq_board_straight(arc_mode, center_middle_rail, camp_corner_straight, a, b, c))


def new_rng_state(seed: int) -> np.ndarray:
    st = np.zeros(4, np.uint64)
    lib().jq_rng_seed(_ptr(st, ctypes.c_uint64), ctypes.c_uint64(seed & (2**64 - 1)))
    return st


def generate_setup(kind: int, rng_state: np.ndarray) -> np.ndarray:
    out = np.full(30, -1, np.int8)
    lib().jq_generate_setup(kind, _ptr(rng_state, ctypes.c_uint64), _ptr(out, ctypes.c_int8))
    return out


def set_setup_params(slot: int, params: np.ndarray) -> None:
    """设置第 slot 组参数化布阵（setup_kind = SETUP_CUSTOM + slot），参数含义见 junqi.h 的 kSetupParams。"""
    p = np.ascontiguousarray(params, dtype=np.float32)
    lib().jq_set_setup_params(slot, _ptr(p, ctypes.c_float))


def valid_setup(camp30: np.ndarray) -> bool:
    a = np.ascontiguousarray(camp30, dtype=np.int8)
    return bool(lib().jq_valid_setup(_ptr(a, ctypes.c_int8)))


class Game:
    """一局四国军旗。节点、座位编号见 engine/include/junqi/junqi.h。"""

    def __init__(self, cfg: RuleConfig | None = None):
        self.cfg = cfg if cfg is not None else make_config()
        self._L = lib()
        self._h = self._L.jq_game_new(ctypes.byref(self.cfg))
        d = dims()
        self.F, self.G = d["node_features"], d["global_features"]
        self._mv = np.zeros(2048, np.int16)

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._L.jq_game_free(h)
            self._h = None

    def reset(self, seed: int = 0, setups: np.ndarray | None = None, setup_kinds=None) -> "Game":
        sp = None
        if setups is not None:
            self._setups_buf = np.ascontiguousarray(setups, dtype=np.int8).reshape(120)
            sp = _ptr(self._setups_buf, ctypes.c_int8)
        kp = None
        if setup_kinds is not None:
            self._kinds_buf = np.ascontiguousarray(setup_kinds, dtype=np.int32)
            kp = _ptr(self._kinds_buf, ctypes.c_int32)
        self._L.jq_game_reset(self._h, ctypes.c_uint64(seed & (2**64 - 1)), sp, kp)
        return self

    def load_position(self, types: np.ndarray, owners: np.ndarray, current_seat: int, seed: int = 0) -> bool:
        """载入任意局面（绝对节点编号）。用于残局课程与规则对拍。"""
        self._pos_buf = (np.ascontiguousarray(types, dtype=np.int8), np.ascontiguousarray(owners, dtype=np.int8))
        return bool(self._L.jq_game_load_position(self._h, _ptr(self._pos_buf[0], ctypes.c_int8),
                                                  _ptr(self._pos_buf[1], ctypes.c_int8), int(current_seat),
                                                  ctypes.c_uint64(seed)))

    # ---- 状态
    @property
    def current_seat(self) -> int:
        return self._L.jq_game_current_seat(self._h)

    @property
    def over(self) -> bool:
        return bool(self._L.jq_game_over(self._h))

    @property
    def winner(self) -> int:
        return self._L.jq_game_winner(self._h)

    @property
    def end_reason(self) -> int:
        return self._L.jq_game_end_reason(self._h)

    @property
    def ply(self) -> int:
        return self._L.jq_game_ply(self._h)

    def team_of(self, seat: int) -> int:
        return self._L.jq_game_team_of(self._h, seat)

    def seat_alive(self, seat: int) -> bool:
        return bool(self._L.jq_game_seat_alive(self._h, seat))

    def flag_revealed(self, seat: int) -> bool:
        return bool(self._L.jq_game_flag_revealed(self._h, seat))

    def legal_moves(self) -> np.ndarray:
        n = self._L.jq_game_legal_moves(self._h, _ptr(self._mv, ctypes.c_int16), 1024)
        return self._mv[: 2 * min(n, 1024)].reshape(-1, 2).astype(np.int64)

    def step(self, frm: int, to: int) -> int:
        return self._L.jq_game_step(self._h, int(frm), int(to))

    # ---- 观测（以 observer 为 0 号座位的相对编号）
    def encode(self, observer: int, nf=None, gf=None, legal=None, want_legal: bool = True):
        nf = np.empty((129, self.F), np.float32) if nf is None else nf
        gf = np.empty(self.G, np.float32) if gf is None else gf
        if want_legal and legal is None:
            legal = np.empty((129, 129), np.uint8)
        self._L.jq_game_encode(
            self._h,
            observer,
            _ptr(nf, ctypes.c_float),
            _ptr(gf, ctypes.c_float),
            _ptr(legal, ctypes.c_uint8) if want_legal else None,
        )
        return nf, gf, legal

    def true_types(self, observer: int, out=None) -> np.ndarray:
        """特权信息，仅供训练标签使用。"""
        out = np.empty(129, np.int8) if out is None else out
        self._L.jq_game_true_types(self._h, observer, _ptr(out, ctypes.c_int8))
        return out

    def board(self):
        t = np.empty(129, np.int8)
        o = np.empty(129, np.int8)
        self._L.jq_game_board(self._h, _ptr(t, ctypes.c_int8), _ptr(o, ctypes.c_int8))
        return t, o

    def belief(self, observer: int) -> np.ndarray:
        out = np.empty(129, np.uint16)
        self._L.jq_game_belief(self._h, observer, _ptr(out, ctypes.c_uint16))
        return out

    def setups(self) -> np.ndarray:
        out = np.empty(120, np.int8)
        self._L.jq_game_setups(self._h, _ptr(out, ctypes.c_int8))
        return out

    def history(self):
        n = self._L.jq_game_history(self._h, None, None, 0)
        mv = np.empty((max(n, 1), 2), np.int16)
        oc = np.empty(max(n, 1), np.int8)
        self._L.jq_game_history(self._h, _ptr(mv, ctypes.c_int16), _ptr(oc, ctypes.c_int8), n)
        return mv[:n].astype(np.int64), oc[:n].astype(np.int64)

    def agent_act(self, kind: int, param: float, rng_state: np.ndarray) -> tuple[int, int]:
        out = np.zeros(2, np.int16)
        self._L.jq_agent_act(self._h, kind, param, _ptr(rng_state, ctypes.c_uint64), _ptr(out, ctypes.c_int16))
        return int(out[0]), int(out[1])


def selfplay(
    cfg: RuleConfig,
    n_games: int,
    seed: int = 0,
    threads: int = 0,
    path: str | Path | None = None,
    append: bool = False,
    agent_kind=(AGENT_RULE,) * 4,
    agent_param=(0.1,) * 4,
    setup_kind=(SETUP_HEURISTIC,) * 4,
) -> dict:
    """C++ 多线程自对弈。返回胜负统计；给出 path 时把棋谱写入 .jqr。"""
    spec = SelfPlaySpec()
    spec.cfg = cfg
    for s in range(4):
        spec.agent_kind[s] = int(agent_kind[s])
        spec.agent_param[s] = float(agent_param[s])
        spec.setup_kind[s] = int(setup_kind[s])
    stats = np.zeros(4, np.int64)
    threads = threads or os.cpu_count() or 1
    p = str(path).encode() if path is not None else None
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    nbytes = lib().jq_selfplay(ctypes.byref(spec), n_games, ctypes.c_uint64(seed), threads, p, int(append),
                               _ptr(stats, ctypes.c_int64))
    if nbytes < 0:
        raise OSError(f"failed to write {path}")
    return dict(team0=int(stats[0]), team1=int(stats[1]), draws=int(stats[2]), plies=int(stats[3]), bytes=int(nbytes))
