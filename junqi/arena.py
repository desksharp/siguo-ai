"""对战场：两个智能体按队伍对打（座位轮换），统计胜率与 Elo 差。

智能体接口（Python 侧）：
    agent.act_batch(games: list[Game], seats: list[int]) -> list[tuple[int, int]]
内置 C++ 智能体（随机 / 规则）走 C++ 多线程快速通道。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import AGENT_EXPLOIT, AGENT_RANDOM, AGENT_RULE, SETUP_HEURISTIC, Game, RuleConfig, new_rng_state, selfplay


@dataclass
class CppAgent:
    kind: int = AGENT_RULE
    param: float = 0.1
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = {AGENT_RANDOM: "random", AGENT_RULE: f"rule(T={self.param:g})",
                         AGENT_EXPLOIT: f"exploit(T={self.param:g})"}.get(self.kind, "cpp")
        self._rng = new_rng_state(hash(self.name) & 0xFFFFFFFF)

    def act_batch(self, games, seats):
        return [g.agent_act(self.kind, self.param, self._rng) for g in games]


def random_agent() -> CppAgent:
    return CppAgent(AGENT_RANDOM, 0.0)


def rule_agent(temperature: float = 0.1) -> CppAgent:
    return CppAgent(AGENT_RULE, temperature)


def exploit_agent(temperature: float = 0.1) -> CppAgent:
    """规则 AI 变体：认定对方后两排没动过的子多半是地雷，专打动过的后排子（惩罚暴露）。"""
    return CppAgent(AGENT_EXPLOIT, temperature)


@dataclass
class MatchResult:
    a: str
    b: str
    games: int
    wins: int
    losses: int
    draws: int

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / max(self.games, 1)

    @property
    def elo(self) -> float:
        return elo_from_score(self.score)

    @property
    def elo_ci95(self) -> tuple[float, float]:
        """按得分的正态近似给出 95% 置信区间。"""
        n = max(self.games, 1)
        p = self.score
        var = (self.wins * (1 - p) ** 2 + self.losses * p**2 + self.draws * (0.5 - p) ** 2) / n
        se = math.sqrt(max(var, 1e-12) / n)
        return elo_from_score(p - 1.96 * se), elo_from_score(p + 1.96 * se)

    def __str__(self) -> str:
        lo, hi = self.elo_ci95
        return (f"{self.a} vs {self.b}: {self.games} games  W/L/D={self.wins}/{self.losses}/{self.draws}  "
                f"score={self.score:.3f}  Elo {self.elo:+.0f} [{lo:+.0f}, {hi:+.0f}]")


def elo_from_score(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return 400.0 * math.log10(p / (1 - p))


def _seats_of_team(cfg: RuleConfig, team: int) -> list[int]:
    return ([0] if team == 0 else [2]) if cfg.mode == 1 else ([0, 2] if team == 0 else [1, 3])


def match_cpp(cfg: RuleConfig, a: CppAgent, b: CppAgent, n_games: int = 1000, seed: int = 0, threads: int = 0,
              setup_kind: int = SETUP_HEURISTIC) -> MatchResult:
    """C++ 快速通道：一半对局 A 执队伍 0，另一半执队伍 1。"""
    half = n_games // 2
    w = l = d = 0
    for a_team, s in ((0, seed), (1, seed + 7919)):
        kinds, params = [0] * 4, [0.0] * 4
        for team, ag in ((a_team, a), (1 - a_team, b)):
            for seat in _seats_of_team(cfg, team):
                kinds[seat], params[seat] = ag.kind, ag.param
        st = selfplay(cfg, half, seed=s, threads=threads, agent_kind=kinds, agent_param=params,
                      setup_kind=[setup_kind] * 4)
        w += st[f"team{a_team}"]
        l += st[f"team{1 - a_team}"]
        d += st["draws"]
    return MatchResult(a.name, b.name, 2 * half, w, l, d)


def play_games(cfg: RuleConfig, agent_a, agent_b, n_games: int = 200, seed: int = 0, batch: int = 64,
               setup_kind: int = SETUP_HEURISTIC, name_a: str = "A", name_b: str = "B") -> MatchResult:
    """通用对战（支持 Python/神经网络智能体）：同时推进 batch 局，按当前行动方分组批量决策。"""
    w = l = d = 0
    rng = np.random.default_rng(seed)
    done_games = 0
    while done_games < n_games:
        k = min(batch, n_games - done_games)
        games, a_team = [], []
        for i in range(k):
            g = Game(cfg).reset(int(rng.integers(2**62)), setup_kinds=[setup_kind] * 4)
            games.append(g)
            a_team.append((done_games + i) % 2)
        live = list(range(k))
        while live:
            groups: dict[int, list[int]] = {0: [], 1: []}
            for i in live:
                g = games[i]
                groups[0 if g.team_of(g.current_seat) == a_team[i] else 1].append(i)
            for who, idx in groups.items():
                if not idx:
                    continue
                agent = agent_a if who == 0 else agent_b
                moves = agent.act_batch([games[i] for i in idx], [games[i].current_seat for i in idx])
                for i, (frm, to) in zip(idx, moves):
                    if games[i].step(frm, to) < 0:
                        raise RuntimeError(f"agent {'A' if who == 0 else 'B'} played illegal move {frm}->{to}")
            live = [i for i in live if not games[i].over]
        for i, g in enumerate(games):
            if g.winner < 0:
                d += 1
            elif g.winner == a_team[i]:
                w += 1
            else:
                l += 1
        done_games += k
    return MatchResult(getattr(agent_a, "name", name_a), getattr(agent_b, "name", name_b), n_games, w, l, d)
