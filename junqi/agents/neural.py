"""神经网络智能体：批量编码各局当前行动方的观测，一次前向，按合法着掩码选着。"""

from __future__ import annotations

import numpy as np
import torch

from ..engine import board_info
from ..model import JunqiNet, masked_log_softmax

N = 129


class NeuralAgent:
    def __init__(self, model: JunqiNet, device: str = "cpu", temperature: float = 0.0, name: str = "neural",
                 seed: int = 0):
        self.model = model.to(device).eval()
        self.device = device
        self.temperature = temperature
        self.name = name
        self.rng = np.random.default_rng(seed)
        c = model.cfg
        self.abs_of = board_info(c.arc_mode, c.center_middle_rail, c.camp_corner_straight).abs_of

    @torch.no_grad()
    def forward(self, games, seats):
        k = len(games)
        F_, G_ = games[0].F, games[0].G
        node = np.empty((k, N, F_), np.float32)
        glob = np.empty((k, G_), np.float32)
        legal = np.empty((k, N, N), np.uint8)
        for i, (g, s) in enumerate(zip(games, seats)):
            g.encode(s, node[i], glob[i], legal[i])
        t = lambda a: torch.from_numpy(a).to(self.device)  # noqa: E731
        logits, belief, value = self.model(t(node), t(glob))
        logp = masked_log_softmax(logits, t(legal.reshape(k, -1)))
        return logp, belief, value

    def act_batch(self, games, seats, prefer=None, bonus: float = 0.0):
        """prefer[i]：这一家优先攻击的敌方座位（None 不偏好），攻击它的着法对数概率加 bonus。"""
        logp, _, _ = self.forward(games, seats)
        if prefer is not None:
            for i, (g, s, ts) in enumerate(zip(games, seats, prefer)):
                if ts is not None:
                    dest = torch.from_numpy(np.flatnonzero(g.board()[1][self.abs_of[s]] == ts)).to(self.device)
                    logp[i].view(N, N)[:, dest] += bonus
        if self.temperature <= 0:
            idx = logp.argmax(-1).cpu().numpy()
        else:
            p = torch.softmax(logp / self.temperature, -1).cpu().numpy().astype(np.float64)
            idx = np.array([self.rng.choice(p.shape[1], p=row / row.sum()) for row in p])
        out = []
        for i, s in zip(idx, seats):
            fr, to = divmod(int(i), N)
            out.append((int(self.abs_of[s][fr]), int(self.abs_of[s][to])))
        return out

    def belief_probs(self, game, observer) -> np.ndarray:
        """[129,12] 绝对编号下的神经信念分布（供 metrics.belief_report 使用）。"""
        _, belief, _ = self.forward([game], [observer])
        p = torch.softmax(belief[0].float(), -1).cpu().numpy()
        out = np.empty_like(p)
        out[self.abs_of[observer]] = p
        return out
