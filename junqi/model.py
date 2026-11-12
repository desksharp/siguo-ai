"""JunqiNet：节点级 Transformer（129 个棋盘点 + 1 个全局 token），注意力带棋盘关系偏置。

输出头
  policy  [B,129,129]  先选子再选落点等价的双线性打分 q_from · k_to，按合法着掩码归一化
  belief  [B,129,12]   每个点上棋子的兵种分布（与硬约束掩码相乘，保证不违反已知约束）
  value   [B,3]        行动方队伍 胜/和/负
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import board_info, dims

N = 129
MASK_SLICE = slice(15, 27)  # 节点特征中的信念掩码位


@dataclass
class NetConfig:
    d_model: int = 192
    layers: int = 6
    heads: int = 6
    ff_mult: int = 4
    dropout: float = 0.0
    arc_mode: int = 2
    center_middle_rail: int = 1
    camp_corner_straight: int = 0


def relation_ids(cfg: NetConfig) -> torch.Tensor:
    """[130,130] 关系编号：距离分桶(0..6) × 边类型(无/公路/铁路)，全局 token 单独一类。"""
    B = board_info(cfg.arc_mode, cfg.center_middle_rail, cfg.camp_corner_straight)
    ek = B.edge_kind.astype(np.int64)
    dist = np.zeros((N, N), np.int64)
    # 无阻挡最短步数（BFS）
    adj = [np.flatnonzero(ek[i]) for i in range(N)]
    for s in range(N):
        d = np.full(N, 99)
        d[s] = 0
        q = [s]
        for a in q:
            for b in adj[a]:
                if d[b] == 99:
                    d[b] = d[a] + 1
                    q.append(b)
        dist[s] = d
    bucket = np.minimum(dist, 6)
    rel = bucket * 3 + ek
    out = np.full((N + 1, N + 1), 21, np.int64)
    out[:N, :N] = rel
    return torch.from_numpy(out)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class JunqiNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or NetConfig()
        dd = dims()
        d = cfg.d_model
        self.inp = nn.Linear(dd["node_features"], d)
        self.node_emb = nn.Parameter(torch.zeros(N, d))
        self.glob = nn.Linear(dd["global_features"], d)
        self.register_buffer("rel", relation_ids(cfg), persistent=False)
        self.rel_bias = nn.Embedding(22, cfg.heads * cfg.layers)
        nn.init.zeros_(self.rel_bias.weight)
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.ff_mult, cfg.dropout) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(d)
        self.pq = nn.Linear(d, d)
        self.pk = nn.Linear(d, d)
        self.belief = nn.Linear(d, 12)
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        nn.init.normal_(self.node_emb, std=0.02)

    def forward(self, node: torch.Tensor, glob: torch.Tensor):
        b = node.shape[0]
        x = self.inp(node) + self.node_emb
        g = self.glob(glob).unsqueeze(1)
        x = torch.cat([x, g], 1)
        L, H = self.cfg.layers, self.cfg.heads
        bias = self.rel_bias(self.rel).view(N + 1, N + 1, L, H).permute(2, 3, 0, 1)  # [L,H,T,T]
        for i, blk in enumerate(self.blocks):
            x = blk(x, bias[i])
        x = self.ln(x)
        h, hg = x[:, :N], x[:, N]
        logits = torch.einsum("bfd,btd->bft", self.pq(h), self.pk(h)) / math.sqrt(self.cfg.d_model)
        mask_bits = node[:, :, MASK_SLICE]
        belief = self.belief(h).float().masked_fill(mask_bits < 0.5, -1e4)
        return logits.reshape(b, N * N), belief, self.value(hg)

    def config_dict(self) -> dict:
        return asdict(self.cfg)


def masked_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float().masked_fill(legal == 0, -1e9), dim=-1)


def losses(model: JunqiNet, batch: dict, w_belief: float = 0.5, w_value: float = 0.5) -> tuple[torch.Tensor, dict]:
    logits, belief, value = model(batch["node"], batch["glob"])
    logp = masked_log_softmax(logits, batch["legal"])
    act = batch["action"]
    l_pi = -logp.gather(1, act[:, None]).squeeze(1).mean()
    bt = batch["btype"].long()
    sel = bt >= 0
    if sel.any():
        l_b = F.cross_entropy(belief[sel], bt[sel])
        b_acc = (belief[sel].argmax(-1) == bt[sel]).float().mean()
    else:
        l_b = belief.sum() * 0
        b_acc = torch.zeros(())
    l_v = F.cross_entropy(value.float(), batch["value"])
    loss = l_pi + w_belief * l_b + w_value * l_v
    with torch.no_grad():
        stats = {
            "loss": loss.item(),
            "policy_nll": l_pi.item(),
            "policy_top1": (logp.argmax(-1) == act).float().mean().item(),
            "belief_nll": l_b.item(),
            "belief_top1": float(b_acc),
            "value_nll": l_v.item(),
            "value_acc": (value.argmax(-1) == batch["value"]).float().mean().item(),
        }
    return loss, stats


def save(model: JunqiNet, path, extra: dict | None = None) -> None:
    torch.save({"config": model.config_dict(), "state": model.state_dict(), **(extra or {})}, path)


def load(path, device="cpu") -> JunqiNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = JunqiNet(NetConfig(**ck["config"]))
    m.load_state_dict(ck["state"])
    return m.to(device).eval()
