"""M3 强化学习：从行为克隆模型出发，用 PPO 自对弈微调，目标是超过规则 AI（老师）。

  python3 -m junqi.train_rl --init runs/bc_four_v4/model.pt --out runs/rl_four_v1 --rules siguo_standard

每一轮（iteration）：
  1. 采样：同时推进 --slots 局棋，一局结束立即在同一槽位开新局。每局随机选对手：
       self  四家都是当前模型（四家的着法都进训练）
       pool  一队是当前模型，另一队是历史快照（起始快照就是 --init 的模型）
       rule  一队是当前模型，另一队是规则 AI（T=0.1 或 T=0）
     只有当前模型走的着法进训练，奖励在终局给：本队胜 +1、负 -1、和 --draw-reward。
     可选的中途奖励（默认关闭，--shape-anneal 轮内线性降到 0）：
       --elim-reward   敌方一家全军覆没时本队 +r、被灭的一队 -r（鼓励两人集火先吃掉一家）
       --reveal-penalty 后两排的子走到空位、而对手此前还认为它可能是地雷时 -p（鼓励装雷）
     --undo-penalty 把自己上一步刚走的子原路走回时 -p（压住来回拖棋）；不随 --shape-anneal 衰减，
       因为拖棋正是在其它中途奖励降到 0 之后变多的（rl_four_v12–v14 来回走子 37% → 45%）
     --p-exploit：对规则 AI 的对局里，这个比例换成“专打暴露后排”的规则 AI 变体。
  2. 复盘：采样时只记着法和概率，不存观测（省内存）；训练前按种子与布阵复盘重新编码。
  3. 更新：PPO（剪切比率）+ 价值头（胜/和/负交叉熵）+ 信念头（辅助）+ 对初始模型的 KL 约束。

每 --eval-every 轮用贪心策略（T=0）对规则 AI 和初始模型各打 --eval-games 局，存 model.pt 并加入快照池。
中断后加 --resume 从 ckpt.pt 接着跑。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .agents.neural import NeuralAgent
from .arena import exploit_agent, play_games, random_agent, rule_agent
from .engine import AGENT_EXPLOIT, AGENT_RULE, SETUP_CUSTOM, SETUP_HEURISTIC, Game, board_info, dims, load_rules, new_rng_state, set_setup_params
from .model import JunqiNet, load, masked_log_softmax, save
from .setup_search import ab_score
from .style import style_report

N = 129
SETUP_KINDS = [SETUP_HEURISTIC] * 4
MINE_BIT = 1 << 1


@dataclass
class RulePolicy:
    temperature: float
    kind: int = AGENT_RULE

    @property
    def name(self) -> str:
        return f"{'rule' if self.kind == AGENT_RULE else 'exploit'}(T={self.temperature:g})"


@dataclass
class NetPolicy:
    model: JunqiNet
    name: str


LEARNER = "L"


@dataclass
class Episode:
    seed: int
    setups: np.ndarray
    roles: list  # 每个座位：LEARNER / RulePolicy / NetPolicy
    opp: str  # self / pool / rule
    learner_team: int  # 自对弈为 -1
    moves: list = field(default_factory=list)
    seats: list = field(default_factory=list)
    logp: list = field(default_factory=list)  # 只有当前模型的着法有值，其余为 nan
    value: list = field(default_factory=list)
    version: list = field(default_factory=list)
    rew: list = field(default_factory=list)  # 中途奖励（记在得奖一方最近一步上），终局奖励另算
    alive: list = field(default_factory=lambda: [True] * 4)
    n_reveal: int = 0
    n_elim: int = 0
    n_undo: int = 0  # 当前模型原路走回的步数
    adv: np.ndarray | None = None
    label: np.ndarray | None = None  # 价值头标签 0=胜 1=和 2=负
    winner: int = -1
    plies: int = 0

    def learner_plies(self) -> np.ndarray:
        return np.flatnonzero(~np.isnan(np.asarray(self.logp, np.float64)))


def scalar_value(value_logits: torch.Tensor, draw_reward: float) -> torch.Tensor:
    p = torch.softmax(value_logits.float(), -1)
    return p[:, 0] - p[:, 2] + draw_reward * p[:, 1]


class Rollout:
    """--slots 局并行推进；当前模型与快照在 GPU 上批量决策，规则 AI 走 C++。"""

    def __init__(self, cfg, model: JunqiNet, pool: list[NetPolicy], a, device: str):
        self.cfg, self.model, self.pool, self.a, self.device = cfg, model, pool, a, device
        self.games = [Game(cfg) for _ in range(a.slots)]
        B = board_info(cfg.arc_mode, cfg.center_middle_rail, cfg.camp_corner_straight)
        self.abs_of = B.abs_of
        # 各家后两排（不含大本营）：地雷只能摆在这里
        self.back = np.stack([(B.seat == s) & (B.row >= 4) & ~B.is_base.astype(bool) for s in range(4)])
        d = dims()
        self.node = np.empty((a.slots, N, d["node_features"]), np.float32)
        self.glob = np.empty((a.slots, d["global_features"]), np.float32)
        self.legal = np.empty((a.slots, N, N), np.uint8)
        self.rng = np.random.default_rng(a.seed + 1)
        self.rule_rng = new_rng_state(a.seed + 2)
        self.amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(a.amp)
        w = np.array([a.p_self, a.p_pool, a.p_rule], np.float64)
        self.opp_p = w / w.sum()
        self.version = 0
        self.done: list[Episode] = []
        self.eps: list[Episode] = [None] * a.slots  # type: ignore[list-item]
        for i in range(a.slots):
            self._new(i)

    def _new(self, i: int) -> None:
        g = self.games[i]
        seed = int(self.rng.integers(2**62))
        kinds = SETUP_KINDS
        if self.a.setup_mix > 0:  # 两队各抽各的：既有同布阵的对局，也有候选对现行
            cand = self.rng.random(2) < self.a.setup_mix
            kinds = [SETUP_CUSTOM if cand[s % 2] else SETUP_HEURISTIC for s in range(4)]
        g.reset(seed, setup_kinds=kinds)
        opp = ("self", "pool", "rule")[self.rng.choice(3, p=self.opp_p)]
        if opp == "pool" and not self.pool:
            opp = "self"
        if opp == "self":
            roles, team = [LEARNER] * 4, -1
        else:
            team = int(self.rng.integers(2))
            if opp == "pool":
                # 偏向最近的快照，但初始模型始终有机会
                k = len(self.pool)
                w = np.arange(1, k + 1, dtype=np.float64)
                other = self.pool[self.rng.choice(k, p=w / w.sum())]
            else:
                other = RulePolicy(0.1 if self.rng.random() < 0.7 else 0.0,
                                   AGENT_EXPLOIT if self.rng.random() < self.a.p_exploit else AGENT_RULE)
            roles = [LEARNER if g.team_of(s) == team else other for s in range(4)]
        self.eps[i] = Episode(seed, g.setups(), roles, opp, team, alive=[g.seat_alive(s) for s in range(4)])

    def shaping(self) -> float:
        """中途奖励的系数：--shape-anneal 轮内从 1 线性降到 0（0 表示一直是 1）。"""
        n = self.a.shape_anneal
        return 1.0 if n <= 0 else max(0.0, 1.0 - (self.version - 1) / n)

    @torch.no_grad()
    def _net_act(self, model: JunqiNet, idx: list[int]):
        k = len(idx)
        for j, i in enumerate(idx):
            g = self.games[i]
            g.encode(g.current_seat, self.node[j], self.glob[j], self.legal[j])
        dev = self.device
        node = torch.from_numpy(self.node[:k]).to(dev, non_blocking=True)
        glob = torch.from_numpy(self.glob[:k]).to(dev, non_blocking=True)
        legal = torch.from_numpy(self.legal[:k].reshape(k, -1)).to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.split(":")[0], dtype=self.amp, enabled=self.amp is not None):
            logits, _, value = model(node, glob)
        logp = masked_log_softmax(logits.float() / self.a.tau, legal)
        gumbel = -torch.log(-torch.log(torch.rand_like(logp).clamp_(1e-20, 1.0)))
        act = (logp + gumbel).argmax(-1)
        lp = logp.gather(1, act[:, None]).squeeze(1)
        v = scalar_value(value, self.a.draw_reward)
        return act.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()

    def tick(self) -> None:
        groups: dict[int, list[int]] = defaultdict(list)
        role_of: dict[int, object] = {}
        for i, g in enumerate(self.games):
            r = self.eps[i].roles[g.current_seat]
            groups[id(r)].append(i)
            role_of[id(r)] = r
        moves: dict[int, tuple] = {}
        for key, idx in groups.items():
            role = role_of[key]
            if isinstance(role, RulePolicy):
                for i in idx:
                    fr, to = self.games[i].agent_act(role.kind, role.temperature, self.rule_rng)
                    moves[i] = (fr, to, math.nan, math.nan)
                continue
            model = self.model if role == LEARNER else role.model
            act, lp, v = self._net_act(model, idx)
            for j, i in enumerate(idx):
                s = self.games[i].current_seat
                fr, to = divmod(int(act[j]), N)
                rec = (float(lp[j]), float(v[j])) if role == LEARNER else (math.nan, math.nan)
                moves[i] = (int(self.abs_of[s][fr]), int(self.abs_of[s][to]), *rec)
        k = self.shaping()
        reveal, elim = self.a.reveal_penalty * k, self.a.elim_reward * k
        for i, (fr, to, lp, v) in moves.items():
            g, ep = self.games[i], self.eps[i]
            s = g.current_seat
            r = 0.0
            if reveal > 0 and not math.isnan(lp) and self.back[s, fr] and g.board()[1][to] < 0:
                foe = next(x for x in range(4) if g.team_of(x) != g.team_of(s) and ep.alive[x])
                if g.belief(foe)[fr] & MINE_BIT:
                    r -= reveal
                    ep.n_reveal += 1
            if not math.isnan(lp) and self._last_move(ep, s) == (to, fr):
                r -= self.a.undo_penalty
                ep.n_undo += 1
            ep.moves.append((fr, to))
            ep.seats.append(s)
            ep.logp.append(lp)
            ep.value.append(v)
            ep.version.append(self.version)
            ep.rew.append(r)
            oc = g.step(fr, to)
            if oc < 0:
                raise RuntimeError(f"illegal move {fr}->{to} by {ep.roles[s]}")
            if oc > 0 and not g.over:
                for x in range(4):
                    if ep.alive[x] and not g.seat_alive(x):
                        ep.alive[x] = False
                        ep.n_elim += 1
                        if elim > 0:
                            self._team_reward(ep, g, g.team_of(x), -elim)
                            self._team_reward(ep, g, 1 - g.team_of(x), elim)
            if g.over:
                self._finish(i)

    @staticmethod
    def _last_move(ep: Episode, s: int) -> tuple | None:
        """座位 s 在这局里上一步走的 (起点, 终点)；四家轮流走，往回看几步就能找到。"""
        for j in range(len(ep.seats) - 1, -1, -1):
            if ep.seats[j] == s:
                return ep.moves[j]
        return None

    @staticmethod
    def _team_reward(ep: Episode, g: Game, team: int, r: float) -> None:
        """给某队每个当前模型座位最近走的那一步记上奖励。"""
        for s in range(4):
            if g.team_of(s) != team or ep.roles[s] != LEARNER:
                continue
            for j in range(len(ep.seats) - 1, -1, -1):
                if ep.seats[j] == s:
                    ep.rew[j] += r
                    break

    def _finish(self, i: int) -> None:
        g, ep, a = self.games[i], self.eps[i], self.a
        ep.winner, ep.plies = g.winner, g.ply
        seats = np.asarray(ep.seats)
        values = np.asarray(ep.value, np.float64)
        ep.adv = np.zeros(len(seats), np.float32)
        ep.label = np.full(len(seats), -1, np.int64)
        for s in range(4):
            if ep.roles[s] != LEARNER:
                continue
            idx = np.flatnonzero(seats == s)
            if len(idx) == 0:
                continue
            if ep.winner < 0:
                z, lab = a.draw_reward, 1
            elif g.team_of(s) == ep.winner:
                z, lab = 1.0, 0
            else:
                z, lab = -1.0, 2
            v = values[idx]
            nxt = np.append(v[1:], z)
            delta = np.asarray(ep.rew)[idx] + nxt - v
            adv = np.zeros(len(idx))
            run = 0.0
            for t in range(len(idx) - 1, -1, -1):
                run = delta[t] + a.lam * run
                adv[t] = run
            ep.adv[idx] = adv
            ep.label[idx] = lab
        self.done.append(ep)
        self._new(i)

    def collect(self, target: int, version: int) -> tuple[list[Episode], int, float]:
        """推进到已完成对局里有至少 target 个可用的当前模型样本为止。"""
        self.version = version
        t0 = time.time()
        out: list[Episode] = []
        n = 0
        while n < target:
            self.tick()
            for ep in self.done:
                lp = ep.learner_plies()
                n += int(np.sum(np.asarray(ep.version)[lp] >= version - self.a.max_lag))
                out.append(ep)
            self.done = []
        return out, n, time.time() - t0


def build_batch(episodes: list[Episode], cfg, version: int, max_lag: int) -> dict:
    """按种子与布阵复盘，为当前模型的着法重新编码观测（与采样时逐位一致）。"""
    B = board_info(cfg.arc_mode, cfg.center_middle_rail, cfg.camp_corner_straight)
    total = 0
    for ep in episodes:
        lp = ep.learner_plies()
        total += int(np.sum(np.asarray(ep.version)[lp] >= version - max_lag))
    d = dims()
    out = {
        "node": np.empty((total, N, d["node_features"]), np.float32),
        "glob": np.empty((total, d["global_features"]), np.float32),
        "legal": np.empty((total, N, N), np.uint8),
        "action": np.empty(total, np.int64),
        "logp": np.empty(total, np.float32),
        "adv": np.empty(total, np.float32),
        "value": np.empty(total, np.int64),
        "btype": np.empty((total, N), np.int8),
    }
    g = Game(cfg)
    m = 0
    for ep in episodes:
        g.reset(ep.seed, ep.setups)
        for p, (fr, to) in enumerate(ep.moves):
            if not math.isnan(ep.logp[p]) and ep.version[p] >= version - max_lag:
                s = g.current_seat
                node, _, legal = g.encode(s, out["node"][m], out["glob"][m], out["legal"][m])
                act = int(B.rel[s][fr]) * N + int(B.rel[s][to])
                if not legal.reshape(-1)[act]:
                    raise RuntimeError("replay mismatch: recorded move is not legal")
                tt = g.true_types(s)
                known = node[:, 31] > 0.5
                occupied = node[:, 8] < 0.5
                out["btype"][m] = np.where(occupied & ~known, tt, -1)
                out["action"][m] = act
                out["logp"][m] = ep.logp[p]
                out["adv"][m] = ep.adv[p]
                out["value"][m] = ep.label[p]
                m += 1
            g.step(fr, to)
    out["legal"] = out["legal"].reshape(total, -1)
    return out


def ppo_update(model: JunqiNet, ref: JunqiNet | None, opt, data: dict, a, device: str) -> dict:
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(a.amp)
    M = len(data["action"])
    adv_all = data["adv"]
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
    rng = np.random.default_rng()
    stats: dict[str, list] = defaultdict(list)
    model.train()
    for epoch in range(a.epochs):
        perm = rng.permutation(M)
        kls = []
        for s in range(0, M, a.minibatch):
            ix = np.sort(perm[s:s + a.minibatch])
            t = lambda k: torch.from_numpy(data[k][ix]).to(device, non_blocking=True)  # noqa: E731
            node, glob, legal, act = t("node"), t("glob"), t("legal"), t("action")
            old, label, btype = t("logp"), t("value"), t("btype").long()
            adv = torch.from_numpy(adv_all[ix]).to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=amp, enabled=amp is not None):
                logits, belief, value = model(node, glob)
                if ref is not None and a.kl_ref > 0:
                    with torch.no_grad():
                        ref_logits = ref(node, glob)[0]
            logp = masked_log_softmax(logits.float() / a.tau, legal)
            lp = logp.gather(1, act[:, None]).squeeze(1)
            ratio = torch.exp(lp - old)
            pg = -torch.min(ratio * adv, ratio.clamp(1 - a.clip, 1 + a.clip) * adv).mean()
            p = logp.exp()
            ent = -(p * logp).sum(-1).mean()
            loss = pg - a.ent * ent
            if ref is not None and a.kl_ref > 0:
                ref_logp = masked_log_softmax(ref_logits.float() / a.tau, legal)
                kl_ref = (p * (logp - ref_logp)).sum(-1).mean()
                loss = loss + a.kl_ref * kl_ref
                stats["kl_ref"].append(kl_ref.item())
            l_v = F.cross_entropy(value.float(), label)
            sel = btype >= 0
            l_b = F.cross_entropy(belief[sel], btype[sel]) if sel.any() else belief.sum() * 0
            loss = loss + a.w_value * l_v + a.w_belief * l_b
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt.step()
            with torch.no_grad():
                approx_kl = (old - lp).mean().item()
                kls.append(approx_kl)
                stats["pg_loss"].append(pg.item())
                stats["entropy"].append(ent.item())
                stats["value_nll"].append(l_v.item())
                stats["value_acc"].append((value.argmax(-1) == label).float().mean().item())
                stats["belief_nll"].append(l_b.item())
                stats["approx_kl"].append(approx_kl)
                stats["clipfrac"].append(((ratio - 1).abs() > a.clip).float().mean().item())
                stats["grad_norm"].append(float(gn))
        stats["epochs"].append(epoch + 1)
        if a.target_kl > 0 and np.mean(kls) > a.target_kl:
            break
    model.eval()
    out = {k: float(np.mean(v)) for k, v in stats.items() if k != "epochs"}
    out["epochs"] = int(stats["epochs"][-1])
    return out


def evaluate(cfg, model: JunqiNet, ref: JunqiNet, a, device: str, seed: int) -> dict:
    # 强化学习后的策略更“混合”，贪心（T=0）容易来回走到无碰子和棋，所以评估默认带一点温度
    agent = NeuralAgent(model, device, temperature=a.eval_temp, name="rl", seed=seed)
    out = {}
    opps = [("vs_rule", rule_agent(0.1)), ("vs_bc", NeuralAgent(ref, device, temperature=a.eval_temp, name="bc", seed=seed + 1))]
    for key, opp in opps:
        r = play_games(cfg, agent, opp, a.eval_games, seed=seed, batch=min(a.eval_games, 256))
        out[key] = r.score
        out[key + "_wld"] = [r.wins, r.losses, r.draws]
    if a.style_games > 0:
        out["style"] = style_report(cfg, agent, a.style_games, seed=seed, batch=min(a.style_games, 256))
    if a.setup_mix > 0:
        out["setup_ab"] = setup_ab(cfg, agent, a.eval_games, seed)
    return out


def setup_ab(cfg, agent, n: int, seed: int) -> dict:
    """候选布阵一队对现行布阵一队的得分。"""
    w, d, g = ab_score(cfg, agent, SETUP_CUSTOM, SETUP_HEURISTIC, n, seed, batch=min(n, 256))
    s = (w + 0.5 * d) / g
    return {"score": s, "ci95": 1.96 * (s * (1 - s) / g) ** 0.5, "games": g}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, required=True, help="起始模型（行为克隆或上一轮强化学习的 model.pt）")
    ap.add_argument("--ref", type=Path, default=None, help="KL 约束与评估用的参照模型，默认同 --init")
    ap.add_argument("--pool-add", type=Path, nargs="*", default=[], help="额外放进对手池、一直保留的模型")
    ap.add_argument("--eval-opp", type=Path, default=None, help="定期评估里“对起点模型”一项换成这个模型（默认用参照模型）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rules", default="siguo_standard")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--slots", type=int, default=512, help="同时进行的对局数")
    ap.add_argument("--samples", type=int, default=32768, help="每轮训练样本数（当前模型的着法）")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95, help="GAE λ（按同一座位的相邻着法）")
    ap.add_argument("--tau", type=float, default=1.0, help="策略温度：采样与训练都用 softmax(logits/τ)")
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--kl-ref", type=float, default=0.02, help="对初始模型的 KL 约束系数，0 关闭")
    ap.add_argument("--target-kl", type=float, default=0.03, help="一轮内新旧策略 KL 超过就提前结束本轮")
    ap.add_argument("--w-value", type=float, default=0.5)
    ap.add_argument("--w-belief", type=float, default=0.25)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--draw-reward", type=float, default=0.0)
    ap.add_argument("--p-self", type=float, default=0.5)
    ap.add_argument("--p-pool", type=float, default=0.25)
    ap.add_argument("--p-rule", type=float, default=0.25)
    ap.add_argument("--p-exploit", type=float, default=0.0, help="对规则 AI 的对局里换成“专打暴露后排”变体的比例")
    ap.add_argument("--elim-reward", type=float, default=0.0, help="敌方一家全军覆没：本队 +r，被灭一队 -r")
    ap.add_argument("--reveal-penalty", type=float, default=0.0, help="后两排可能是雷的子走到空位：-p")
    ap.add_argument("--undo-penalty", type=float, default=0.0, help="把上一步刚走的子原路走回：-p（不随 --shape-anneal 衰减）")
    ap.add_argument("--shape-anneal", type=int, default=0, help="中途奖励在这么多轮内线性降到 0，0 不衰减")
    ap.add_argument("--style-games", type=int, default=0, help="每次评估附带风格体检的局数（四家自对弈），0 关闭")
    ap.add_argument("--setup-params", type=Path, default=None, help="候选布阵的 JSON（setup_search 输出，含 params）")
    ap.add_argument("--setup-mix", type=float, default=0.0, help="每队用候选布阵的概率，其余用现行布阵；评估附带候选对现行的得分")
    ap.add_argument("--pool-size", type=int, default=6, help="对手池大小（含参照模型和 --pool-add）")
    ap.add_argument("--max-lag", type=int, default=3, help="采样时的模型最多落后几轮，再旧的着法丢弃")
    ap.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--eval-temp", type=float, default=0.5, help="评估时双方神经网络的走子温度，0 为贪心")
    ap.add_argument("--final-games", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="从 out/ckpt.pt 接着跑")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "snapshots").mkdir(exist_ok=True)
    cfg = load_rules(a.rules)
    dev = a.device
    if a.setup_mix > 0:
        if not a.setup_params:
            raise SystemExit("--setup-mix 需要 --setup-params")
        set_setup_params(0, np.array(json.loads(a.setup_params.read_text())["params"]))

    ref = load(a.ref or a.init, dev)
    ref.requires_grad_(False)
    model = load(a.init, dev)
    extra = [load(p, dev).requires_grad_(False) for p in a.pool_add]
    eval_opp = load(a.eval_opp, dev).requires_grad_(False) if a.eval_opp else ref
    for m in [model, ref, eval_opp, *extra]:
        c = m.cfg
        if (c.arc_mode, c.center_middle_rail, c.camp_corner_straight) != (
                cfg.arc_mode, cfg.center_middle_rail, cfg.camp_corner_straight):
            raise SystemExit("起始/参照/对手池模型的棋盘规则与 --rules 不一致")
    c = model.cfg
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99), eps=1e-6)
    pool = [NetPolicy(ref, "init")] + [NetPolicy(m, f"fixed{i}") for i, m in enumerate(extra)]
    n_fixed = len(pool)
    start = 1
    if a.resume and (a.out / "ckpt.pt").exists():
        ck = torch.load(a.out / "ckpt.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        start = ck["iter"] + 1
        for p in ck["pool"][n_fixed - 1:] if ck.get("n_fixed") else ck["pool"]:
            pool.append(NetPolicy(load(a.out / "snapshots" / p, dev).requires_grad_(False), p))
        # 丢掉检查点之后的日志（那几轮会重跑），换随机种子避免重复同样的对局
        lines = (a.out / "log.jsonl").read_text(encoding="utf-8").splitlines() if (a.out / "log.jsonl").exists() else []
        keep = [s for s in lines if s.strip() and json.loads(s).get("iter", 0) <= ck["iter"]]
        (a.out / "log.jsonl").write_text("".join(s + "\n" for s in keep), encoding="utf-8")
        a.seed += 7919 * start
        print(f"resumed at iter {start}", flush=True)
    elif (a.out / "log.jsonl").exists() and not a.resume:
        raise SystemExit(f"{a.out} 已有训练记录：换一个目录，或加 --resume 接着跑")
    else:
        (a.out / "log.jsonl").unlink(missing_ok=True)  # --resume 但还没存过检查点：从头开始
        argv = [x for x in sys.argv[1:] if x != "--resume"]
        (a.out / "args.json").write_text(json.dumps(
            {**vars(a), "algo": "ppo",
             "params": n_params, "d_model": c.d_model, "layers": c.layers, "heads": c.heads, "steps": a.iters,
             "argv": argv, "board": {k: getattr(cfg, k) for k in ("arc_mode", "center_middle_rail", "camp_corner_straight")}},
            indent=2, ensure_ascii=False, default=str))
    print(f"params: {n_params / 1e6:.2f}M  device: {dev}  slots: {a.slots}  samples/iter: {a.samples}", flush=True)
    log = open(a.out / "log.jsonl", "a", encoding="utf-8")

    if start == 1:
        ev = evaluate(cfg, model, eval_opp, a, dev, seed=10_000)
        log.write(json.dumps({"iter": 0, "step": 0, "eval": ev}) + "\n")
        log.flush()
        print(f"  [eval] iter 0  {ev}", flush=True)

    roll = Rollout(cfg, model, pool, a, dev)
    t_start = time.time()
    for it in range(start, a.iters + 1):
        t0 = time.time()
        eps, n_avail, t_roll = roll.collect(a.samples, it)
        t1 = time.time()
        data = build_batch(eps, cfg, it, a.max_lag)
        t2 = time.time()
        st = ppo_update(model, ref, opt, data, a, dev)
        t3 = time.time()

        res: dict[str, list] = defaultdict(lambda: [0, 0, 0])
        for ep in eps:
            if ep.learner_team < 0:
                continue
            key = "rule" if ep.opp == "rule" else "pool"
            r = res[key]
            r[0 if ep.winner == ep.learner_team else 2 if ep.winner < 0 else 1] += 1
        rec = {
            "iter": it, "step": it, "samples": len(data["action"]), "games": len(eps),
            "plies": float(np.mean([e.plies for e in eps])),
            "draw_rate": float(np.mean([e.winner < 0 for e in eps])),
            "rollout_s": t_roll, "replay_s": t2 - t1, "train_s": t3 - t2,
            "samples_per_s": len(data["action"]) / (t3 - t0), "lr": a.lr,
            "elapsed_s": time.time() - t_start, **st,
            "elims_per_game": float(np.mean([e.n_elim for e in eps])),
            "reveals_per_learner_seat": sum(e.n_reveal for e in eps) / max(sum(r == LEARNER for e in eps for r in e.roles), 1),
            "undo_rate": sum(e.n_undo for e in eps) / max(len(data["action"]), 1),
            "shaping": roll.shaping(),
        }
        for k, (w, l, d) in res.items():
            rec[f"roll_{k}"] = (w + 0.5 * d) / max(w + l + d, 1)
            rec[f"roll_{k}_n"] = w + l + d
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"iter {it:4d}  games {len(eps):4d}  samples {rec['samples']:6d}  plies {rec['plies']:.0f}  "
              f"pg {st['pg_loss']:+.4f}  ent {st['entropy']:.3f}  kl_ref {st.get('kl_ref', 0):.4f}  "
              f"akl {st['approx_kl']:.4f}  clip {st['clipfrac']:.3f}  v_acc {st['value_acc']:.3f}  "
              + "  ".join(f"{k} {rec[f'roll_{k}']:.3f}/{rec[f'roll_{k}_n']}" for k in sorted(res))
              + f"  [{t_roll:.0f}s+{t2 - t1:.0f}s+{t3 - t2:.0f}s]", flush=True)
        del data

        if it % a.eval_every == 0 or it == a.iters:
            ev = evaluate(cfg, model, eval_opp, a, dev, seed=10_000 + it)
            log.write(json.dumps({"iter": it, "step": it, "eval": ev}) + "\n")
            log.flush()
            print(f"  [eval] iter {it}  {ev}", flush=True)
            save(model, a.out / "model.pt", {"step": it, "iter": it, "algo": "ppo"})
            name = f"it{it:05d}.pt"
            save(model, a.out / "snapshots" / name, {"iter": it})
            snap = load(a.out / "snapshots" / name, dev).requires_grad_(False)
            pool.append(NetPolicy(snap, name))
            if len(pool) > a.pool_size:
                pool.pop(n_fixed)  # 参照模型和 --pool-add 的模型始终保留
            torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "iter": it, "n_fixed": n_fixed,
                        "pool": [p.name for p in pool[1:]]}, a.out / "ckpt.pt")

    # ---- 结束评估
    agent = NeuralAgent(model, dev, temperature=a.eval_temp, name="rl", seed=1)
    summary: dict = {"params": n_params, "steps": a.iters, "algo": "ppo", "init": str(a.init), "eval_temp": a.eval_temp}
    opps = [random_agent(), rule_agent(0.1), rule_agent(0.0), exploit_agent(0.1),
            NeuralAgent(eval_opp, dev, temperature=a.eval_temp, name="bc", seed=2)]
    for opp in opps:
        n = a.eval_games if opp.name == "random" else a.final_games
        r = play_games(cfg, agent, opp, n, seed=123, batch=256)
        print(r, flush=True)
        summary[f"vs_{opp.name}"] = {"score": r.score, "elo": r.elo, "w": r.wins, "l": r.losses, "d": r.draws}
    summary["style"] = style_report(cfg, agent, max(a.style_games, 300), seed=123, batch=256)
    print("style", summary["style"], flush=True)
    if a.setup_mix > 0:
        summary["setup_ab"] = setup_ab(cfg, agent, a.final_games, 123)
        print("setup_ab", summary["setup_ab"], flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
