"""对局分析（神经网络部分），只用人类玩家这一方看得到的信息。

  evaluate()  人类视角一次前向：本队胜/和/负、每颗暗子的兵种分布；轮到人类时顺带给出 AI 首选的几步
  predict()   轮到别家时：按人类视角的神经信念采样若干个完整局面（人类自己的子用真值，其余暗子按分布抽，
              每家不超编制），站在该家视角前向，把各局面的着法分布平均，作为“它下一步会怎么走”的预测。
              采样局面从头开始推断（看不到真实对局里该家积累的信息），所以是偏保守的近似，但不含任何隐藏信息。
"""

from __future__ import annotations

import numpy as np
import torch

from .engine import QUOTA, Game

N = 129
BITS = np.arange(12)


class Analyst:
    def __init__(self, agent, cfg, samples: int = 8, top: int = 3):
        self.agent = agent  # NeuralAgent
        self.cfg = cfg
        self.samples = samples
        self.top = top
        self.abs_of = agent.abs_of

    def _top_moves(self, p: np.ndarray, seat: int) -> list[dict]:
        out = []
        for i in np.argsort(-p)[:self.top]:
            if p[i] < 0.02:
                break
            fr, to = divmod(int(i), N)
            out.append({"from": int(self.abs_of[seat][fr]), "to": int(self.abs_of[seat][to]), "p": round(float(p[i]), 3)})
        return out

    def evaluate(self, game: Game, human: int) -> dict:
        logp, belief, value = self.agent.forward([game], [human])
        wdl = torch.softmax(value[0].float(), -1).numpy()
        rel = torch.softmax(belief[0].float(), -1).numpy()
        probs = np.empty_like(rel)
        probs[self.abs_of[human]] = rel
        out = {"wdl": [round(float(x), 3) for x in wdl], "probs": probs}
        if not game.over and game.current_seat == human:
            out["moves"] = self._top_moves(torch.softmax(logp[0], -1).numpy(), human)
        return out

    def predict(self, game: Game, human: int, probs: np.ndarray, rng: np.random.Generator) -> list[dict]:
        seat = game.current_seat
        types, owners = game.board()
        masks = game.belief(human)
        worlds = []
        for _ in range(self.samples):
            t = types.copy()
            for s in range(4):
                if s == human:
                    continue
                rem = QUOTA.copy()
                hidden = []
                for n in map(int, np.flatnonzero(owners == s)):
                    m = int(masks[n])
                    if m and not m & (m - 1):
                        rem[m.bit_length() - 1] -= 1
                        t[n] = m.bit_length() - 1
                    else:
                        hidden.append(n)
                hidden.sort(key=lambda n: bin(int(masks[n])).count("1"))  # 候选少的先抽，避免后面无子可选
                for n in hidden:
                    cand = (int(masks[n]) >> BITS) & 1
                    w = probs[n] * cand * (rem > 0)
                    if w.sum() <= 0:
                        w = probs[n] * cand + 1e-9 * cand
                    k = int(rng.choice(12, p=w / w.sum()))
                    t[n] = k
                    rem[k] -= 1
            g = Game(self.cfg)
            if g.load_position(t, owners, seat, int(rng.integers(1 << 30))) and not g.over:
                worlds.append(g)
        if not worlds:
            return []
        logp, _, _ = self.agent.forward(worlds, [seat] * len(worlds))
        return self._top_moves(torch.softmax(logp, -1).mean(0).numpy(), seat)
