"""规则回归测试：手写的走法用例 + 引擎自对弈快照。

  1. 走法用例：空棋盘上摆一颗子，检查它能走到哪些点（弧线、第 5 排直角、工兵拐弯、九宫中路）。
  2. 自对弈快照：固定种子的规则 AI 对局，逐步核对布阵、合法着法集合、碰子结果和终局。
     规则或走法生成器的任何改动都会让快照对不上；确认改动是有意的之后，用 --regen 重新生成。

运行：python3 tests/test_regression.py [--regen]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from junqi.engine import AGENT_RULE, Game, load_rules, make_config, new_rng_state  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "selfplay_regression.json"
FLAG, ENGINEER, COMPANY = 0, 3, 5
SOUTH, EAST, NORTH, WEST = 0, 1, 2, 3
# (规则, 种子)：四国暗棋 6 局 + 二人 2 局
CASES = [("siguo_standard", s) for s in range(6)] + [("duel", s) for s in range(2)]

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)


def node(seat: int, row: int, col: int) -> int:
    return seat * 30 + row * 5 + col


# ---------------------------------------------------------------- 走法用例

def reach(piece: int, at: int, cfg=None) -> set[int]:
    """南家一颗子单独在棋盘上（其他三家只在大本营留一面军旗）时能走到的点。"""
    types = np.full(129, -1, np.int8)
    owners = np.full(129, -1, np.int8)
    types[at], owners[at] = piece, SOUTH
    for s in (EAST, NORTH, WEST):
        types[node(s, 5, 1)], owners[node(s, 5, 1)] = FLAG, s
    g = Game(cfg or make_config())
    assert g.load_position(types, owners, SOUTH)
    return {int(b) for a, b in g.legal_moves() if a == at}


def side_rail(seat: int, col: int) -> set[int]:
    return {node(seat, r, col) for r in range(5)}


def test_moves():
    # 边线上的非工兵：沿边线过弧线进入西家边线，但不能拐进西家第 5 排，也到不了东家
    r = reach(COMPANY, node(SOUTH, 2, 0))
    check(side_rail(WEST, 4) <= r, "连长应能沿边线过弧线进入西家边线")
    check(node(WEST, 4, 3) not in r, "连长不能在西家第 5 排拐角拐弯")
    check(not any(n // 30 == EAST for n in r if n < 120), "连长不应到达东家")

    # 前排中间的非工兵：沿前排走到角点，但从前排进弧线算拐弯
    r = reach(COMPANY, node(SOUTH, 0, 2))
    check(node(SOUTH, 0, 0) in r and node(SOUTH, 0, 4) in r, "连长应能沿前排走到两端")
    check(not any(n // 30 in (EAST, WEST) for n in r if n < 120), "连长不能从前排进弧线")
    check(node(NORTH, 0, 2) in r, "连长应能经九宫中路直达北家前排中点")

    # 第 5 排上的非工兵：沿第 5 排走到边线端点后不能拐上边线
    r = reach(COMPANY, node(SOUTH, 4, 2))
    check(node(SOUTH, 4, 0) in r and node(SOUTH, 4, 4) in r, "连长应能沿第 5 排走到两端")
    check(node(SOUTH, 2, 0) not in r and node(SOUTH, 2, 4) not in r, "连长不能在第 5 排直角拐上边线")

    # 工兵在铁路上任意拐弯：同一位置能到四家全部铁路点
    r = reach(ENGINEER, node(SOUTH, 0, 2))
    for s in (EAST, NORTH, WEST):
        check(side_rail(s, 0) | side_rail(s, 4) <= r, f"工兵应能到达座位 {s} 的两条边线")
    check(set(range(120, 129)) <= r, "工兵应能到达九宫全部点")

    # 规则开关：允许第 5 排直角直行（camp_corner_straight=1）后，同一颗连长能拐上边线
    r = reach(COMPANY, node(SOUTH, 4, 2), make_config(camp_corner_straight=1))
    check(node(SOUTH, 2, 0) in r, "camp_corner_straight=1 时连长应能拐上边线")


# ---------------------------------------------------------------- 自对弈快照

def play(rules: str, seed: int) -> dict:
    """规则 AI（T=0.1）四家对局。legal 是每步合法着法集合的摘要。"""
    g = Game(load_rules(rules)).reset(seed)
    rng = new_rng_state(seed)
    setups = g.setups().tolist()
    moves, outcomes, legal = [], [], hashlib.sha256()
    while not g.over:
        lm = g.legal_moves()
        legal.update(np.ascontiguousarray(lm[np.lexsort((lm[:, 1], lm[:, 0]))], np.int16).tobytes())
        frm, to = g.agent_act(AGENT_RULE, 0.1, rng)
        moves.append([frm, to])
        outcomes.append(g.step(frm, to))
    return {"rules": rules, "seed": seed, "setups": setups, "moves": moves, "outcomes": outcomes,
            "legal_sha256": legal.hexdigest(), "winner": g.winner, "end_reason": g.end_reason, "plies": g.ply}


def replay(rec: dict) -> list[str]:
    """按记录的布阵和着法逐步复盘，报告第一处不一致。"""
    tag = f"{rec['rules']} seed {rec['seed']}"
    g = Game(load_rules(rec["rules"]))
    if g.reset(rec["seed"]).setups().tolist() != rec["setups"]:
        return [f"{tag}: 布阵生成结果变了"]
    g.reset(rec["seed"], setups=np.array(rec["setups"], np.int8))
    for i, ((frm, to), oc) in enumerate(zip(rec["moves"], rec["outcomes"])):
        if not any(int(a) == frm and int(b) == to for a, b in g.legal_moves()):
            return [f"{tag}: 第 {i} 步 {frm}->{to} 不再合法"]
        got = g.step(frm, to)
        if got != oc:
            return [f"{tag}: 第 {i} 步碰子结果 {got}，记录是 {oc}"]
    errs = []
    if (g.winner, g.end_reason, g.ply) != (rec["winner"], rec["end_reason"], rec["plies"]):
        errs.append(f"{tag}: 终局 {(g.winner, g.end_reason, g.ply)}，记录是 {(rec['winner'], rec['end_reason'], rec['plies'])}")
    now = play(rec["rules"], rec["seed"])
    if now["legal_sha256"] != rec["legal_sha256"]:
        errs.append(f"{tag}: 合法着法集合变了")
    if now["moves"] != rec["moves"]:
        errs.append(f"{tag}: 规则 AI 的着法变了（规则没变时多半是规则 AI 或随机数改了）")
    return errs


def test_selfplay_snapshot():
    if not FIX.exists():
        check(False, f"缺少快照 {FIX.relative_to(ROOT)}，先运行 --regen")
        return
    recs = json.loads(FIX.read_text(encoding="utf-8"))
    check([(r["rules"], r["seed"]) for r in recs] == CASES, "快照里的对局列表与 CASES 不一致")
    for rec in recs:
        for e in replay(rec):
            check(False, e)
    print(f"  {len(recs)} 局，共 {sum(r['plies'] for r in recs)} 步")


def regen():
    recs = [play(rules, seed) for rules, seed in CASES]
    FIX.parent.mkdir(parents=True, exist_ok=True)
    FIX.write_text("[\n" + ",\n".join(json.dumps(r, separators=(",", ":")) for r in recs) + "\n]\n", encoding="utf-8")
    print(f"wrote {FIX.relative_to(ROOT)}: {len(recs)} games, {sum(r['plies'] for r in recs)} plies")


def main() -> int:
    if "--regen" in sys.argv:
        regen()
        return 0
    for t in (test_moves, test_selfplay_snapshot):
        print(t.__name__)
        t()
    print("REGRESSION", "FAILED" if FAILS else "PASSED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
