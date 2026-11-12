"""Python 层测试：棋谱读写与复盘、观测泄漏测试、载入局面、对战场。

运行：python3 tests/test_python.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from junqi import records  # noqa: E402
from junqi.engine import Game, dims, load_rules, make_config, selfplay, valid_setup  # noqa: E402

FAILS: list[str] = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("FAIL:", msg)


def test_records_roundtrip():
    for rules in ("siguo_standard", "duel"):
        cfg = load_rules(rules)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "g.jqr"
            st = selfplay(cfg, 40, seed=7, threads=4, path=path)
            recs = records.read(path)
            check(len(recs) == 40, f"{rules}: record count {len(recs)}")
            check(sum(r.plies for r in recs) == st["plies"], f"{rules}: plies mismatch")
            wins = [sum(r.winner == t for r in recs) for t in (0, 1)]
            check(wins == [st["team0"], st["team1"]], f"{rules}: winners mismatch {wins} vs {st}")
            for r in recs[:10]:
                for s in range(4):
                    camp = r.setups[30 * s:30 * s + 30]
                    if (camp >= 0).any():
                        check(valid_setup(camp), f"{rules}: invalid setup in record")
                g = None
                for g, *_ in records.replay(r):
                    pass
                g.step(*map(int, r.moves[-1]))
                check(g.over and g.winner == r.winner, f"{rules}: replay result mismatch")
                records.write(Path(d) / "copy.jqr", [r], append=True)
            again = records.read(Path(d) / "copy.jqr")
            check(all((a.moves == b.moves).all() and a.seed == b.seed for a, b in zip(again, recs[:10])),
                  f"{rules}: python write/read mismatch")


def _combat_pieces(rec, cfg):
    """复盘一局，返回参与过碰子的节点轨迹集合（按子跟踪）。"""
    g = Game(cfg).reset(rec.seed, rec.setups)
    # 按初始节点跟踪每颗子
    where = {n: n for n in range(129) if g.board()[0][n] >= 0}
    fought = set()
    for frm, to in rec.moves:
        frm, to = int(frm), int(to)
        mover = next(k for k, v in where.items() if v == frm)
        target = next((k for k, v in where.items() if v == to), None)
        g.step(frm, to)
        types, _ = g.board()
        if target is not None:
            fought.update((mover, target))
            where.pop(target)
        where[mover] = to
        # 被移除的（碰子阵亡 / 出局清盘）
        for k in list(where):
            if types[where[k]] < 0 or (k != mover and where[k] == to and types[to] >= 0 and k == target):
                where.pop(k, None)
    return fought


def _replay_outcomes(cfg, rec, setups):
    """公开信息签名：每步碰子结果 + 已亮军旗的位置。"""
    g = Game(cfg).reset(rec.seed, setups)
    out = []
    for frm, to in rec.moves:
        o = g.step(int(frm), int(to))
        if o < 0:
            return None
        types, owners = g.board()
        flags = tuple(int(n) for n in np.flatnonzero(types == 0) if g.flag_revealed(int(owners[n])))
        out.append((o, flags))
    return out


def test_no_leak():
    """交换未碰子的敌子真实身份：只要整局着法仍合法、碰子结果不变（公开信息相同），
    观测必须逐位一致（文档 7.4 泄漏测试）。会覆盖工兵/地雷/炸弹/军官之间的交换。"""
    rng = np.random.default_rng(3)
    for rules in ("siguo_standard", "duel"):
        cfg = load_rules(rules)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "g.jqr"
            selfplay(cfg, 16, seed=11, threads=4, path=path)
            recs = records.read(path)
        tested = 0
        for rec in recs:
            base_out = _replay_outcomes(cfg, rec, rec.setups)
            fought = _combat_pieces(rec, cfg)
            for observer in ((0, 2) if rules == "duel" else (0, 1)):
                g0 = Game(cfg).reset(rec.seed, rec.setups)
                team = g0.team_of(observer)
                for _attempt in range(30):
                    setups = rec.setups.copy()
                    changed = False
                    for s in range(4):
                        if g0.team_of(s) == team or not (setups[30 * s:30 * s + 30] >= 0).any():
                            continue
                        nodes = [s * 30 + p for p in range(30) if setups[s * 30 + p] >= 0 and s * 30 + p not in fought]
                        if len(nodes) < 2:
                            continue
                        a, b = rng.choice(nodes, 2, replace=False)
                        if setups[a] != setups[b]:
                            setups[a], setups[b] = setups[b], setups[a]
                            changed = True
                    if not changed or not all(valid_setup(setups[30 * s:30 * s + 30])
                                              for s in range(4) if (setups[30 * s:30 * s + 30] >= 0).any()):
                        continue
                    if _replay_outcomes(cfg, rec, setups) != base_out:
                        continue  # 公开信息变了（着法非法 / 碰子结果 / 亮旗位置不同），不是合格的对照
                    g0 = Game(cfg).reset(rec.seed, rec.setups)
                    g1 = Game(cfg).reset(rec.seed, setups)
                    for ply, (frm, to) in enumerate(rec.moves):
                        a0, a1 = g0.encode(observer), g1.encode(observer)
                        if not all(np.array_equal(x, y) for x, y in zip(a0, a1)):
                            diff = np.argwhere(a0[0] != a1[0])
                            check(False, f"{rules}: observation leak at ply {ply} observer {observer}: "
                                         f"[node, feature] {diff[:6].tolist()}")
                            break
                        g0.step(int(frm), int(to))
                        g1.step(int(frm), int(to))
                    tested += 1
        print(f"  {rules}: {tested} swapped replays compared")
        check(tested > 0, f"{rules}: leak test did not run")


def test_load_position():
    g = Game(make_config())
    types = np.full(129, -1, np.int8)
    owners = np.full(129, -1, np.int8)
    types[0], owners[0] = 11, 0  # 南营前排左角：司令
    types[30 + 5 * 5 + 1], owners[30 + 5 * 5 + 1] = 0, 1  # 东营大本营：军旗
    types[60 + 5 * 5 + 1], owners[60 + 5 * 5 + 1] = 0, 2
    types[90 + 5 * 5 + 1], owners[90 + 5 * 5 + 1] = 0, 3
    check(g.load_position(types, owners, 0), "load_position failed")
    mv = g.legal_moves()
    check(len(mv) > 10, "marshal on rail should have many moves")
    # 司令沿南北贯通线可以直达北营后排铁路（北营 r4 c4）
    check(any(int(t) == 60 + 4 * 5 + 4 for f, t in mv), "straight rail flight to the far camp")


def test_dims_and_encode():
    d = dims()
    g = Game(load_rules("siguo_standard")).reset(5)
    nf, gf, legal = g.encode(0)
    check(nf.shape == (129, d["node_features"]) and gf.shape == (d["global_features"],), "encode shapes")
    check(int(legal.sum()) == len(g.legal_moves()), "legal mask count == legal moves")
    # 相对编号：自己的 25 子都在相对营区 0 内
    check(nf[:30, 9].sum() == 25, "own pieces in relative camp 0")
    nf1, _, _ = g.encode(1)
    check(nf1[90:120, 9 + 3].sum() == 25, "seat 0 seen from seat 1 is the previous seat (relative camp 3)")


def test_rl_replay():
    """强化学习：采样时记下的着法概率，复盘重新编码后用同一模型算出来必须一致（训练依赖这一点）。"""
    try:
        import torch
    except ImportError:
        print("  (skip: no torch)")
        return
    from types import SimpleNamespace

    from junqi.engine import set_setup_params
    from junqi.model import JunqiNet, NetConfig, masked_log_softmax
    from junqi.setup_search import base_params
    from junqi.train_rl import Rollout, build_batch

    torch.manual_seed(0)
    cfg = load_rules("siguo_standard", max_plies=300)
    model = JunqiNet(NetConfig(d_model=32, layers=1, heads=2)).eval()
    a = SimpleNamespace(slots=6, seed=0, amp="none", tau=1.0, draw_reward=0.0, lam=0.95, max_lag=3,
                        p_self=0.5, p_pool=0.0, p_rule=0.5, p_exploit=0.5,
                        elim_reward=0.2, reveal_penalty=0.05, undo_penalty=0.0, shape_anneal=0, setup_mix=0.5)
    set_setup_params(0, base_params(hq=(1.0, 0.0, 0.0)))  # 候选布阵：另一个大本营一定是雷
    roll = Rollout(cfg, model, [], SimpleNamespace(**{**vars(a), "setup_mix": 1.0}), "cpu")
    hq = np.array([ep.setups for ep in roll.eps]).reshape(-1, 30)[:, [26, 28]]
    check((np.sort(hq, axis=1) == [0, 1]).all(), "rl setup mix: candidate setup used")
    roll = Rollout(cfg, model, [], a, "cpu")
    eps, n, _ = roll.collect(150, version=1)
    data = build_batch(eps, cfg, 1, a.max_lag)
    check(len(data["action"]) == n and n >= 150, "rl replay: sample count")
    with torch.no_grad():
        logits = model(torch.from_numpy(data["node"]), torch.from_numpy(data["glob"]))[0]
        logp = masked_log_softmax(logits, torch.from_numpy(data["legal"]))
        lp = logp.gather(1, torch.from_numpy(data["action"])[:, None]).squeeze(1).numpy()
    check(np.abs(lp - data["logp"]).max() < 1e-3, "rl replay: recorded log-prob matches replayed observation")
    check(set(np.unique(data["value"])) <= {0, 1, 2} and np.isfinite(data["adv"]).all(), "rl replay: labels/advantages")
    # 中途奖励只记在当前模型的着法上；装雷惩罚确实触发过
    for ep in eps:
        lp_nan = np.isnan(np.asarray(ep.logp, np.float64))
        check(len(ep.rew) == len(ep.moves) and not np.any(np.asarray(ep.rew)[lp_nan]), "rl shaping: only learner moves")
    check(sum(ep.n_reveal for ep in eps) > 0, "rl shaping: reveal penalty fired")
    check(any(r.kind == 2 for ep in eps for r in ep.roles if hasattr(r, "kind")) or not any(ep.opp == "rule" for ep in eps),
          "rl shaping: exploit opponent used")
    # 来回走子惩罚：只罚当前模型把上一步原路走回，其余着法不受影响；--shape-anneal 降到 0 后照样罚
    b = SimpleNamespace(**{**vars(a), "elim_reward": 0.0, "reveal_penalty": 0.0, "undo_penalty": 0.1, "shape_anneal": 1})
    roll = Rollout(cfg, model, [], b, "cpu")
    roll.version = 5
    eps, _, _ = roll.collect(150, version=5)
    for ep in eps:
        last, want = {}, []
        for (f, t), s, lp in zip(ep.moves, ep.seats, ep.logp):
            want.append(-0.1 if not np.isnan(lp) and last.get(s) == (t, f) else 0.0)
            last[s] = (f, t)
        check(np.allclose(ep.rew, want), "rl shaping: undo penalty on exactly the learner's undo moves")
        check(ep.n_undo == sum(w < 0 for w in want), "rl shaping: undo count")
    check(sum(ep.n_undo for ep in eps) > 0, "rl shaping: undo penalty fired")


def test_play_analysis_no_leak():
    """人机对弈的分析（胜率、读子分布、走法预测、发给网页的局面）只能依赖人类这一方的信息：
    人类布阵和公开着法相同、另外三家布阵不同的两局，分析结果必须完全一样。"""
    try:
        import torch
    except ImportError:
        print("  (skip: no torch)")
        return
    from junqi.engine import SETUP_HEURISTIC, generate_setup, new_rng_state
    from junqi.model import JunqiNet, NetConfig, save
    from junqi.play import Session

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.pt"
        save(JunqiNet(NetConfig(d_model=32, layers=1, heads=2)).eval(), path)
        rng = new_rng_state(3)
        mine = generate_setup(SETUP_HEURISTIC, rng).tolist()
        sessions = []
        for _ in range(2):
            s = Session(path)
            s.rng = new_rng_state(9)
            setups = {str(k): generate_setup(SETUP_HEURISTIC, rng).tolist() for k in (1, 2, 3)}
            s.new_game(0, {"0": mine, **setups}, {1: "rule", 2: "rule", 3: "rule"})
            sessions.append(s)
        a, b = sessions
        # 人类走一步不碰子的棋（两局都合法），轮到东家：此时会做走法预测
        quiet = lambda s: {(int(f), int(t)) for f, t in s.game.legal_moves() if s.game.board()[1][t] < 0}  # noqa: E731
        mv = sorted(quiet(a) & quiet(b))[0]
        for s in sessions:
            s.human_move(*mv)
        for i in range(2):
            fa, fb = a.frames[i], b.frames[i]
            check(fa["wdl"] == fb["wdl"], f"play analysis: win rate depends on hidden pieces (frame {i})")
            check(fa["moves"] == fb["moves"], f"play analysis: move prediction depends on hidden pieces (frame {i})")
            check(fa["probs"].keys() == fb["probs"].keys() and
                  all(np.allclose(fa["probs"][n], fb["probs"][n], atol=1e-6) for n in fa["probs"]),
                  f"play analysis: belief depends on hidden pieces (frame {i})")
            check(a.frame(i) == b.frame(i), f"play analysis: review frame differs (frame {i})")
        check(bool(a.frames[1]["moves"]), "play analysis: no prediction for the next seat")
        sa, sb = a.state(), b.state()
        check(sa["pieces"] == sb["pieces"] and sa["timeline_tail"] == sb["timeline_tail"], "play analysis: state differs")


def test_focus_preference():
    """集火偏好：只看公开的“队友刚打了谁”；加分很大时一定打那一家。"""
    try:
        import torch
    except ImportError:
        print("  (skip: no torch)")
        return
    from junqi.agents.neural import NeuralAgent
    from junqi.engine import AGENT_RULE, SETUP_HEURISTIC, generate_setup, new_rng_state
    from junqi.model import JunqiNet, NetConfig, save
    from junqi.play import FOCUS_WINDOW, Session

    torch.manual_seed(0)
    model = JunqiNet(NetConfig(d_model=32, layers=1, heads=2)).eval()
    agent = NeuralAgent(model, "cpu")
    cfg = load_rules("siguo_standard")
    rng = new_rng_state(5)
    g = Game(cfg).reset(5)
    hits = 0
    while hits < 5 and not g.over:
        s, owners = g.current_seat, g.board()[1]
        foes = {int(owners[t]) for _, t in g.legal_moves() if owners[t] >= 0}
        for ts in foes:
            f, t = agent.act_batch([g], [s], [ts], 1e6)[0]
            check(owners[t] == ts, "focus: large bonus attacks the preferred seat")
            hits += 1
        g.step(*g.agent_act(AGENT_RULE, 0.5, rng))
    check(hits >= 5, "focus: found positions to test")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.pt"
        save(model, path)
        sess = Session(path)
        sess.new_game(0, {"0": generate_setup(SETUP_HEURISTIC, rng).tolist()}, {1: "rule", 2: "rule", 3: "rule"})
        sess.events = [{"seat": 2, "target_seat": 1}, {"seat": 1, "target_seat": None}]
        check(sess._focus_target(0) == 1, "focus: follows the partner's last target")
        check(sess._focus_target(1) is None, "focus: nothing to follow without the partner's attack")
        sess.events += [{"seat": 3, "target_seat": None}] * FOCUS_WINDOW
        check(sess._focus_target(0) is None, "focus: old attacks are ignored")


def test_play_replay():
    """复盘：存盘的对局按布阵和着法重放，每一帧都和对局时的真实局面一致。"""
    from junqi import play
    from junqi.engine import SETUP_CUSTOM, SETUP_HEURISTIC, generate_setup, new_rng_state, set_setup_params
    from junqi.setup_search import base_params

    with tempfile.TemporaryDirectory() as d:
        old, play.GAMES = play.GAMES, Path(d)
        try:
            s = play.Session(Path(d) / "no_model.pt")
            s.rng = new_rng_state(5)
            s.new_game(0, {"0": generate_setup(SETUP_HEURISTIC, s.rng).tolist()}, {1: "rule", 2: "rule", 3: "rule"})
            rng = new_rng_state(5)
            generate_setup(SETUP_HEURISTIC, rng)
            set_setup_params(0, base_params())
            ai = [generate_setup(SETUP_CUSTOM, rng).tolist() for _ in range(3)]
            check(s.origin["setups"][30:] == sum(ai, []), "play: AI seats use the previous heuristic setup")
            boards = [s.game.board()]
            for _ in range(60):
                if s.game.current_seat == 0:
                    f, t = map(int, s.game.legal_moves()[0])
                    s.human_move(f, t)
                else:
                    s.ai_step()
                boards.append(s.game.board())
            gid = s.save_game()
            r = s.replay(gid)
            check("error" not in r and len(r["frames"]) == 61, f"replay: {r.get('error')} frames {len(r.get('frames', []))}")
            for i, f in enumerate(r.get("frames", [])):
                types, owners = boards[i]
                occ = [[int(n), int(types[n]), int(owners[n])] for n in np.flatnonzero(owners >= 0)]
                if f["p"] != occ:
                    check(False, f"replay: frame {i} differs from the real board")
                    break
            check(play.Session.list_games()["items"][0]["id"] == gid, "replay: saved game not listed")
            check("error" in s.replay("../runs/x"), "replay: bad id accepted")
        finally:
            play.GAMES = old


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            n0 = len(FAILS)
            fn()
            print(f"{name}: {'ok' if len(FAILS) == n0 else 'FAILED'}")
    print("PYTHON TESTS", "FAILED" if FAILS else "PASSED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
