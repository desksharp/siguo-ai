"""布阵搜索：不动模型，只找更好的“各兵种放哪里”的概率。

  python3 -m junqi.setup_search calibrate --nn runs/rl_four_v2/model.pt   # 规则 AI 的判断和神经网络一致吗
  python3 -m junqi.setup_search search --out runs/setup_search            # 进化搜索（规则 AI 对打，只用 CPU）
  python3 -m junqi.setup_search search-nn --nn runs/rl_four_v2/model.pt --out runs/setup_nn  # 神经网络 + 规则 AI 各占一半
  python3 -m junqi.setup_search verify --nn runs/rl_four_v3/model.pt --params runs/setup_search/best.json

都是“同一个智能体四家对打，一队用候选布阵、另一队用基准布阵”，看候选一队的得分。
参数是权重的对数；布阵仍然随机抽，不是固定一套摆法。约束：另一个大本营只放雷/排长/连长，工兵不放最后一排。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .engine import AGENT_EXPLOIT, AGENT_RULE, SETUP_CUSTOM, SETUP_LEGACY, Game, load_rules, selfplay, set_setup_params

# 炸弹…司令的下标 = 兵种编号 - 2；顺序：炸弹 工兵 排 连 营 团 旅 师 军 司令
ROW_BY_TYPE = {  # 上一版启发式（2026-10-31 至 11-06 的默认布阵）的行权重
    0: (0.0, 2.0, 2.0, 1.5, 1.0, 0.3), 1: (1.5, 2.0, 1.5, 1.5, 1.0, 0.0),
    2: (2.0, 1.5, 1.0, 1.0, 1.0, 1.2), 3: (2.0, 1.5, 1.0, 1.0, 1.0, 1.2),
    4: (2.0, 1.5, 1.5, 1.5, 1.0, 0.7), 5: (2.0, 1.5, 1.5, 1.5, 1.0, 0.7),
    6: (2.5, 2.0, 2.0, 1.5, 1.0, 0.5), 7: (2.5, 2.0, 2.0, 1.5, 1.0, 0.5),
    8: (3.0, 2.5, 2.0, 1.5, 1.0, 0.3), 9: (3.0, 2.5, 2.0, 1.5, 1.0, 0.3),
}
HQ_SIZE, MINE_SIZE, ROW_AT, COL_AT, N_PARAMS = 3, 8, 11, 71, 121


def base_params(hq=(0.25, 0.45, 0.30)) -> np.ndarray:
    """上一版启发式（军旗在第 1 列）写成参数，搜索和验证的基准。"""
    p = np.ones(N_PARAMS, np.float64)
    p[:HQ_SIZE] = hq
    p[3:8] = (1.5, 6.0, 1.5, 1.5, 1.5)  # 第 4 行，军旗正前方是护旗位
    p[8:11] = (6.0, 6.0, 2.0)  # 第 5 行第 0、2、4 列，军旗两侧是护旗位
    for t, w in ROW_BY_TYPE.items():
        p[ROW_AT + t * 6: ROW_AT + (t + 1) * 6] = w
    return p


def frozen_mask() -> np.ndarray:
    """不参与搜索的参数：权重为 0 的格子（炸弹不上第 0 行、工兵不放最后一排）。"""
    return base_params() <= 0


def to_params(theta: np.ndarray, base: np.ndarray) -> np.ndarray:
    p = np.where(base > 0, np.exp(theta), 0.0)
    p[:HQ_SIZE] = np.maximum(p[:HQ_SIZE], 1e-3)
    return p


def score_rule(cfg, cand: np.ndarray, base: np.ndarray, n: int, seed: int, threads: int, kinds=(AGENT_RULE, AGENT_EXPLOIT)) -> float:
    """候选布阵对基准布阵：规则 AI 四家对打（两队轮换、两种规则 AI 各一半）。"""
    set_setup_params(0, cand)
    set_setup_params(1, base)
    w = g = 0.0
    q = max(n // (2 * len(kinds)), 1)
    for ki, k in enumerate(kinds):
        for team in (0, 1):
            sk = [SETUP_CUSTOM if s % 2 == team else SETUP_CUSTOM + 1 for s in range(4)]
            st = selfplay(cfg, q, seed=seed + 1000 * ki + team, threads=threads, agent_kind=(k,) * 4, agent_param=(0.1,) * 4, setup_kind=sk)
            w += st[f"team{team}"] + 0.5 * st["draws"]
            g += q
    return w / g


def ab_score(cfg, agent, kind_a: int, kind_b: int, n: int, seed: int, batch: int = 24) -> tuple[int, int, int]:
    """同一个智能体四家对打，一队 kind_a、另一队 kind_b（两队轮换）。返回 kind_a 一队的胜、和、总局数。"""
    import torch

    rng = np.random.default_rng(seed)
    wins = draws = done = 0
    started = 0
    games, team = [], []

    def new():
        nonlocal started
        t = started % 2
        started += 1
        sk = [kind_a if s % 2 == t else kind_b for s in range(4)]
        return Game(cfg).reset(int(rng.integers(2**62)), setup_kinds=sk), t

    for _ in range(min(batch, n)):
        g, t = new()
        games.append(g)
        team.append(t)
    while games:
        with torch.inference_mode():
            mv = agent.act_batch(games, [g.current_seat for g in games])
        keep_g, keep_t = [], []
        for g, t, (f, to) in zip(games, team, mv):
            g.step(f, to)
            if g.over:
                done += 1
                wins += g.winner == t
                draws += g.winner < 0
                if started < n:
                    g, t = new()
                else:
                    continue
            keep_g.append(g)
            keep_t.append(t)
        games, team = keep_g, keep_t
    return wins, draws, done


def _nn_worker(a):
    path, cand, base, n, temp, seed, batch = a
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    from .model import load
    from .agents.neural import NeuralAgent

    torch.set_num_threads(1)
    cfg = load_rules("siguo_standard")
    for slot, p in ((0, cand), (1, base)):
        if isinstance(p, str):
            continue
        set_setup_params(slot, p)
    kind = lambda p, slot: SETUP_LEGACY if isinstance(p, str) else SETUP_CUSTOM + slot  # noqa: E731
    ag = NeuralAgent(load(path, "cpu"), "cpu", temperature=temp, seed=seed)
    return ab_score(cfg, ag, kind(cand, 0), kind(base, 1), n, seed, batch)


def score_nn(path, cand, base, n: int, seed: int, procs: int, temp: float = 0.5, batch: int = 24) -> tuple[float, float, int]:
    """候选布阵对基准布阵：神经网络四家对打（多进程 CPU）。cand/base 可以是 \"legacy\"。"""
    each = n // procs
    with mp.get_context("spawn").Pool(procs) as pool:
        res = pool.map(_nn_worker, [(str(path), cand, base, each, temp, seed + i, batch) for i in range(procs)])
    w, d, g = (sum(r[i] for r in res) for i in range(3))
    s = (w + 0.5 * d) / g
    return s, 1.96 * (s * (1 - s) / g) ** 0.5, g


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def calibrate(a):
    """随机扰动几组布阵，规则 AI 和神经网络各打一遍，看两者的判断是否相关。"""
    cfg = load_rules("siguo_standard")
    base = base_params()
    rng = np.random.default_rng(a.seed)
    th0 = np.log(np.where(base > 0, base, 1.0))
    cands = [("雷占六成", base_params(hq=(0.6, 0.25, 0.15)))]
    for i in range(a.cands):
        cands.append((f"随机{i}", to_params(th0 + a.sigma * rng.standard_normal(N_PARAMS), base)))
    rows = []
    for name, c in cands:
        t = time.time()
        r = score_rule(cfg, c, base, a.rule_games, a.seed, a.threads)
        s, ci, g = score_nn(a.nn, c, base, a.nn_games, a.seed, a.threads)
        rows.append((name, r, s, ci))
        print(f"{name}: rule {r:.3f}  nn {s:.3f} ± {ci:.3f} ({g} 局, {time.time() - t:.0f}s)", flush=True)
    x, y = [r[1] for r in rows], [r[2] for r in rows]
    print(f"Spearman {spearman(x, y):.2f}  Pearson {float(np.corrcoef(x, y)[0, 1]):.2f}  n={len(rows)}", flush=True)


def search(a):
    """(μ/μ, λ) 进化策略，对称采样，候选都和基准布阵比。"""
    cfg = load_rules("siguo_standard")
    base = base_params()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    th0 = np.log(np.where(base > 0, base, 1.0))
    theta = th0.copy()
    live = ~frozen_mask()
    sigma, best = a.sigma, -1.0
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    for gen in range(1, a.gens + 1):
        eps = rng.standard_normal((a.pop // 2, N_PARAMS)) * live
        eps = np.concatenate([eps, -eps])
        f = np.array([score_rule(cfg, to_params(theta + sigma * e, base), base, a.games, a.seed + gen * 100 + i, a.threads)
                      for i, e in enumerate(eps)])
        order = np.argsort(-f)[: len(f) // 2]
        w = np.log(len(order) + 0.5) - np.log(np.arange(1, len(order) + 1))
        theta = theta + sigma * (w / w.sum()) @ eps[order]
        sigma = max(a.sigma_min, sigma * a.sigma_decay)
        rec = {"gen": gen, "best": float(f.max()), "mean": float(f.mean()), "sigma": sigma, "elapsed_s": time.time() - t0}
        if gen % a.eval_every == 0 or gen == a.gens:
            cur = to_params(theta, base)
            rec["eval"] = score_rule(cfg, cur, base, a.eval_games, a.seed + 77, a.threads)
            if rec["eval"] > best:
                best = rec["eval"]
                (out / "best.json").write_text(json.dumps({"gen": gen, "eval": best, "params": cur.tolist()}))
            (out / "latest.json").write_text(json.dumps({"gen": gen, "eval": rec["eval"], "params": cur.tolist()}))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)


def _rival(a, base: np.ndarray) -> np.ndarray:
    """对手布阵：--rival 给的 JSON，默认上一版启发式。"""
    return np.array(json.loads(Path(a.rival).read_text())["params"]) if a.rival else base


def search_nn(a):
    """同样的进化策略，得分 = 神经网络和规则 AI 各一半。

    神经网络是在别的布阵分布上训练的，没见过的摆法可能只是骗过了它学到的先验；规则 AI 没有这种先验，
    但下法和模型不同。两边都赢才算好布阵。
    """
    cfg = load_rules("siguo_standard")
    base = base_params()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    start = np.array(json.loads(Path(a.start).read_text())["params"]) if a.start else base
    rival = _rival(a, base)
    theta = np.log(np.where(base > 0, np.maximum(start, 1e-3), 1.0))
    live = ~frozen_mask()
    sigma, best = a.sigma, -1.0
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    t0 = time.time()

    def both(c, seed, nn_games, rule_games):
        r = score_rule(cfg, c, rival, rule_games, seed, a.threads)
        s, _, _ = score_nn(a.nn, c, rival, nn_games, seed, a.threads)
        return r, s

    for gen in range(1, a.gens + 1):
        eps = rng.standard_normal((a.pop // 2, N_PARAMS)) * live
        eps = np.concatenate([eps, -eps])
        res = [both(to_params(theta + sigma * e, base), a.seed + gen * 1000 + i * 37, a.games, a.rule_games) for i, e in enumerate(eps)]
        f = np.array([(r + s) / 2 for r, s in res])
        order = np.argsort(-f)[: len(f) // 2]
        w = np.log(len(order) + 0.5) - np.log(np.arange(1, len(order) + 1))
        theta = theta + sigma * (w / w.sum()) @ eps[order]
        sigma = max(a.sigma_min, sigma * a.sigma_decay)
        rec = {"gen": gen, "fit_mean": float(f.mean()), "fit_best": float(f.max()), "rule_mean": float(np.mean([r for r, _ in res])),
               "nn_mean": float(np.mean([s for _, s in res])), "sigma": sigma, "elapsed_s": time.time() - t0}
        if gen % a.eval_every == 0 or gen == a.gens:
            cur = to_params(theta, base)
            r, s = both(cur, a.seed + 555, a.eval_nn_games, a.eval_rule_games)
            rec.update(eval_rule=r, eval_nn=s)
            doc = {"gen": gen, "eval_rule": r, "eval_nn": s, "params": cur.tolist()}
            (out / "latest.json").write_text(json.dumps(doc))
            if (r + s) / 2 > best:
                best = (r + s) / 2
                (out / "best.json").write_text(json.dumps(doc))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)


def verify(a):
    base = base_params()
    cand = np.array(json.loads(Path(a.params).read_text())["params"])
    against = (("指定对手", _rival(a, base)),) if a.rival else (("上一版启发式", base), ("旧版布阵", "legacy"))
    for name, b in against[:1] if a.base_only else against:
        s, ci, g = score_nn(a.nn, cand, b, a.games, a.seed, a.threads)
        print(f"候选 vs {name}: {s:.3f} ± {ci:.3f}（{g} 局）", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("calibrate")
    c.add_argument("--nn", required=True)
    c.add_argument("--cands", type=int, default=7)
    c.add_argument("--sigma", type=float, default=0.8)
    c.add_argument("--rule-games", type=int, default=20000)
    c.add_argument("--nn-games", type=int, default=1000)
    s = sp.add_parser("search")
    s.add_argument("--out", required=True)
    s.add_argument("--gens", type=int, default=200)
    s.add_argument("--pop", type=int, default=32)
    s.add_argument("--games", type=int, default=2000)
    s.add_argument("--sigma", type=float, default=0.3)
    s.add_argument("--sigma-decay", type=float, default=0.995)
    s.add_argument("--sigma-min", type=float, default=0.1)
    s.add_argument("--eval-every", type=int, default=10)
    s.add_argument("--eval-games", type=int, default=40000)
    n = sp.add_parser("search-nn")
    n.add_argument("--nn", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--rival", default=None, help="对手布阵的 JSON（含 params），默认上一版启发式")
    n.add_argument("--start", default=None, help="起点布阵的 JSON（含 params），默认上一版启发式")
    n.add_argument("--gens", type=int, default=20)
    n.add_argument("--pop", type=int, default=16)
    n.add_argument("--games", type=int, default=300, help="每个候选的神经网络局数")
    n.add_argument("--rule-games", type=int, default=20000)
    n.add_argument("--sigma", type=float, default=0.3)
    n.add_argument("--sigma-decay", type=float, default=0.98)
    n.add_argument("--sigma-min", type=float, default=0.1)
    n.add_argument("--eval-every", type=int, default=3)
    n.add_argument("--eval-nn-games", type=int, default=1000)
    n.add_argument("--eval-rule-games", type=int, default=40000)
    v = sp.add_parser("verify")
    v.add_argument("--nn", required=True)
    v.add_argument("--params", required=True)
    v.add_argument("--games", type=int, default=2000)
    v.add_argument("--rival", default=None, help="只和这个布阵比（JSON，含 params）")
    v.add_argument("--base-only", action="store_true", help="只对上一版启发式，不对旧版")
    for p in (c, s, n, v):
        p.add_argument("--threads", type=int, default=10)
        p.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    {"calibrate": calibrate, "search": search, "search-nn": search_nn, "verify": verify}[a.cmd](a)


if __name__ == "__main__":
    main()
