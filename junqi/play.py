"""人机对弈：本地网页上和 AI 下四国暗棋（你一家，AI 三家）。

  python3 -m junqi.play                 # 打开 http://localhost:8765
  python3 -m junqi.play --port 9000 --model runs/bc_four_v1/model.pt

网页只拿到你这一方看得到的信息（自己的子、公开的走子与碰子结果、由此推出的候选兵种），
其余棋子的真实身份留在服务端，终局后才公开。
每局结束后整局（四家布阵 + 着法）存到 games/，网页的“复盘”页可以四家明棋 / 暗棋任选地回看。
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .engine import (AGENT_RANDOM, AGENT_RULE, END_REASONS, OUTCOME_NAMES, QUOTA, SETUP_CUSTOM, TYPE_NAMES, Game,
                     board_info, generate_setup, is_straight, load_rules, new_rng_state, set_setup_params, valid_setup)
from .setup_search import base_params

WEB = Path(__file__).resolve().parent / "web"
ENGINEER = TYPE_NAMES.index("工兵")
ROOT = Path(__file__).resolve().parent.parent
SEAT_NAMES = ["南", "东", "北", "西"]
DEFAULT_MODEL = ROOT / "runs" / "bc_four_v1" / "model.pt"
# 走子温度：rl_four_v5 上 0.25 对 0（贪心）0.577、对 0.5 0.548（各 500 局，±0.044）；贪心六成走子是走回原位，四家对打和棋 45%
AI_TEMPERATURE = 0.25
# 集火偏好：神经网络 AI 打队友刚打过的那家时加分。rl_four_v5 上 2.0 让选同一家的比例从 47% 到 56%（随机 48%），
# 胜率基本不变（一队加分对一队不加 0.504 ± 0.044，500 局），只是下法更像在配合
FOCUS_WINDOW = 8
FOCUS_BONUS = 2.0
# 服务端布阵（AI 和“随机布阵”）用上一版启发式：rl_four_v10/v11 上它一队对现行默认一队 0.682/0.666（各 1000 局，±0.03），
# 训练里两种布阵各占一半也没有回落，不是现行布阵被看熟了，是它本身更弱
SETUP = SETUP_CUSTOM
GAMES = ROOT / "games"  # 已结束对局，供复盘
GAME_ID = re.compile(r"^\d{8}-\d{6}(-\d+)?$")
# 子力分值（按 TYPE_NAMES 顺序：军旗 地雷 炸弹 工兵 排长 连长 营长 团长 旅长 师长 军长 司令），满编 88 分
VALUE = np.array([0, 3, 5, 2, 1, 2, 3, 4, 5, 6, 8, 10], np.float64)
BITS = np.arange(12)


class NeuralPlayer:
    """延迟加载 PyTorch，没有 torch 或模型时退回规则 AI。"""

    def __init__(self, path: Path):
        import torch

        from .agents.neural import NeuralAgent
        from .model import load

        torch.set_num_threads(2)
        self.agent = NeuralAgent(load(path, "cpu"), "cpu", temperature=AI_TEMPERATURE, name=path.parent.name)

    def act(self, game: Game, prefer: int | None = None, bonus: float = 0.0) -> tuple[int, int]:
        return self.agent.act_batch([game], [game.current_seat], [prefer], bonus)[0]


class Session:
    def __init__(self, model_path: Path, focus_bonus: float = FOCUS_BONUS):
        self.focus_bonus = focus_bonus
        set_setup_params(SETUP - SETUP_CUSTOM, base_params())
        self.cfg = load_rules("siguo_standard")
        self.info = board_info(self.cfg.arc_mode, self.cfg.center_middle_rail, self.cfg.camp_corner_straight)
        self.model_path = model_path
        self.lock = threading.Lock()
        self.rng = new_rng_state(int(time.time() * 1e6))
        self.game: Game | None = None
        self.human = 0
        self.ai: dict = {}
        self.events: list[dict] = []
        self.frames: list[dict] = []  # 分析帧：frames[i] 是走完 i 步后你这一方看到的局面
        self.origin: dict = {}
        self.god = False  # 上帝视角（调试）：网页显示全部暗子
        self.saved_id: str | None = None  # 本局结束后存盘的编号
        self._neural: NeuralPlayer | None = None
        self._analyst = None
        self._neural_mtime = 0.0
        self._straight_cache: dict = {}

    # ---- AI
    def neural(self) -> NeuralPlayer | None:
        if not self.model_path.exists():
            return None
        mt = self.model_path.stat().st_mtime
        if self._neural is None or mt != self._neural_mtime:  # 训练中会定期覆盖 model.pt，换新权重
            try:
                self._neural, self._neural_mtime = NeuralPlayer(self.model_path), mt
            except Exception as e:  # noqa: BLE001 - 没装 torch 等情况
                print(f"neural model unavailable: {e}")
                return None
        return self._neural

    def analyst(self):
        """神经网络分析器（与 AI 共用同一份权重；没有模型时返回 None，只做规则分析）。"""
        p = self.neural()
        if p is None:
            return None
        if self._analyst is None or self._analyst.agent is not p.agent:
            from .analysis import Analyst
            self._analyst = Analyst(p.agent, self.cfg)
        return self._analyst

    def ai_kinds(self) -> list[dict]:
        out = [{"id": "rule_soft", "name": "规则 AI"}, {"id": "rule", "name": "规则 AI（贪心，易僵持）"},
               {"id": "random", "name": "随机走子"}]
        if self.model_path.exists():
            out.insert(0, {"id": "neural", "name": f"神经网络（{self.model_path.parent.name}）"})
        return out

    def _focus_target(self, seat: int) -> int | None:
        """队友最近攻击过、还没出局的敌方座位（谁攻击谁是公开信息）。"""
        for e in reversed(self.events[-FOCUS_WINDOW:]):
            if e["seat"] == (seat + 2) % 4 and e["target_seat"] is not None and self.game.seat_alive(e["target_seat"]):
                return e["target_seat"]
        return None

    def ai_move(self, seat: int) -> tuple[int, int]:
        kind = self.ai.get(seat, "rule_soft")
        if kind == "neural" and (p := self.neural()) is not None:
            return p.act(self.game, self._focus_target(seat), self.focus_bonus)
        if kind == "random":
            return self.game.agent_act(AGENT_RANDOM, 0.0, self.rng)
        return self.game.agent_act(AGENT_RULE, 0.1 if kind == "rule_soft" else 0.0, self.rng)

    # ---- 对局
    def new_game(self, human: int, setups: dict, ai: dict) -> dict:
        """setups: {座位: 30 个兵种}；没给的座位由服务端随机布阵（调试时可以四家都给）。"""
        arr = np.empty((4, 30), np.int8)
        for s in range(4):
            camp = setups.get(str(s), setups.get(s))
            if camp is None:
                if s == human:
                    return {"error": "缺少你的布阵"}
                arr[s] = generate_setup(SETUP, self.rng)
                continue
            camp = np.array(camp, np.int8)
            if camp.shape != (30,) or not valid_setup(camp):
                return {"error": f"{SEAT_NAMES[s]}家布阵不合规"}
            arr[s] = camp
        origin = {"kind": "setups", "seed": int(self.rng[0]), "setups": arr.reshape(-1).tolist()}
        return self._start(origin, human, ai)

    def load_position(self, types: list[int], owners: list[int], current: int, human: int, ai: dict) -> dict:
        """摆棋：从任意局面开始（绝对节点编号，空位 -1）。"""
        if len(types) != 129 or len(owners) != 129:
            return {"error": "局面数据长度不对"}
        origin = {"kind": "position", "seed": int(self.rng[0]), "types": [int(t) for t in types],
                  "owners": [int(o) for o in owners], "current": int(current)}
        g = self._replay(origin, [])
        if g is None:
            return {"error": "引擎拒绝这个局面（每家最多 25 子；军旗只能在大本营；行棋方必须有子）"}
        if not g.over and len(g.legal_moves()) == 0:
            return {"error": f"{SEAT_NAMES[current]}家没有能走的子"}
        return self._start(origin, human, ai)

    def _replay(self, origin: dict, moves: list) -> Game | None:
        g = Game(self.cfg)
        if origin["kind"] == "setups":
            g.reset(origin["seed"], setups=np.array(origin["setups"], np.int8))
        elif not g.load_position(np.array(origin["types"], np.int8), np.array(origin["owners"], np.int8),
                                 origin["current"], origin["seed"]):
            return None
        for f, t in moves:
            if g.step(f, t) < 0:
                raise RuntimeError(f"replay failed at {f}->{t}")
        return g

    def _start(self, origin: dict, human: int, ai: dict) -> dict:
        self.origin = origin
        self.game = self._replay(origin, [])
        self.human = human
        self.ai = {int(k): v for k, v in ai.items()}
        self.events = []
        self.frames = []
        self.saved_id = None
        self._push_frame()
        return self.state()

    def undo(self) -> dict:
        """悔棋：回到你上一步走之前（中间 AI 的着法一并撤回）。"""
        idx = [i for i, e in enumerate(self.events) if e["seat"] == self.human]
        if self.game is None or not idx:
            return {"error": "没有可以悔的棋"}
        keep = idx[-1]
        self.events = self.events[:keep]
        self.frames = self.frames[:keep + 1]
        self.game = self._replay(self.origin, [(e["from"], e["to"]) for e in self.events])
        self.saved_id = None  # 终局后悔棋再下完，另存一份
        return self.state()

    def hint(self) -> dict:
        """提示：让 AI 站在你的视角（只用你能看到的信息）给一步。"""
        g = self.game
        if g is None or g.over or g.current_seat != self.human:
            return {"error": "现在不是你走"}
        p = self.neural()
        frm, to = p.act(g) if p is not None else g.agent_act(AGENT_RULE, 0.0, self.rng)
        return {"from": frm, "to": to, "by": "neural" if p is not None else "rule"}

    def snapshot(self) -> dict:
        """当前对局的真实局面（调试/摆棋用，含全部暗子）。"""
        if self.game is None:
            return {"error": "还没有对局"}
        types, owners = self.game.board()
        return {"types": types.tolist(), "owners": owners.tolist(), "current": self.game.current_seat}

    def _straight(self, a: int, b: int, c: int) -> bool:
        key = (a, b, c)
        if key not in self._straight_cache:
            cfg = self.cfg
            self._straight_cache[key] = is_straight(a, b, c, cfg.arc_mode, cfg.center_middle_rail, cfg.camp_corner_straight)
        return self._straight_cache[key]

    def _path(self, frm: int, to: int, game: Game | None = None) -> list[int]:
        """走子路线（在走之前调用，供网页沿实际线路播放动画）。
        相邻一步直接连；铁路上非工兵只能走直线，工兵可以拐弯，途经点必须是空的，取最短路线。"""
        ek = self.info.edge_kind
        if ek[frm, to]:
            return [frm, to]
        types, owners = (game or self.game).board()
        engineer = int(types[frm]) == ENGINEER
        start = (frm, -1)
        prev = {start: None}
        q = deque([start])
        while q:
            node, last = cur = q.popleft()
            for nb in map(int, np.flatnonzero(ek[node] == 2)):
                if nb != to and owners[nb] >= 0:
                    continue
                if not engineer and last >= 0 and not self._straight(last, node, nb):
                    continue
                st = (nb, -1 if engineer else node)  # 工兵只看到过哪个点，非工兵还要记来的方向
                if st in prev:
                    continue
                prev[st] = cur
                if nb == to:
                    path = []
                    while st is not None:
                        path.append(st[0])
                        st = prev[st]
                    return path[::-1]
                q.append(st)
        return [frm, to]

    def _record(self, seat: int, frm: int, to: int, oc: int, path: list[int] | None = None) -> dict:
        ev = {"ply": self.game.ply, "seat": seat, "from": frm, "to": to, "outcome": OUTCOME_NAMES[oc],
              "path": path or [frm, to]}
        prev = self.frames[-1]
        cur = self._push_frame()
        cur["ev"] = [seat, ev["outcome"]]
        before = {n: (o, m) for n, o, _, m in prev["view"]}
        ev["i"] = len(self.frames) - 1
        ev["mover"] = cand_list(before[frm][1])
        ev["target"] = cand_list(before[to][1]) if to in before else None
        ev["target_seat"] = before[to][0] if to in before else None
        ev["dmat"] = [round(float(a - b), 1) for a, b in zip(cur["sum"]["mat"], prev["sum"]["mat"])]
        # 这一步让哪些暗子的候选变少了（进攻方活下来时，目标点上的子就是原来出发点上的那颗）
        narrowed = []
        for n, o, _, m in cur["view"]:
            src = frm if n == to and oc in (0, 1, 4) else n
            if src in before and bin(m).count("1") < bin(before[src][1]).count("1"):
                narrowed.append({"node": n, "seat": o, "cand": cand_list(m), "gone": cand_list(before[src][1] & ~m)})
        ev["narrowed"] = narrowed
        # 这一步在上一帧的预测（别家）或 AI 首选（你）里排第几；0 = 不在前几位
        if prev.get("moves") is not None:
            rank = next((k + 1 for k, mv in enumerate(prev["moves"]) if mv["from"] == frm and mv["to"] == to), 0)
            ev["rank"], ev["rank_kind"] = rank, "suggest" if seat == self.human else "predict"
            ev["rank_p"] = prev["moves"][rank - 1]["p"] if rank else None
            cur["hit"] = rank
        if "wdl" in prev and "wdl" in cur:
            ev["dwin"] = round(cur["wdl"][0] - prev["wdl"][0], 3)  # 你方胜率变化
        self.events.append(ev)
        if self.game.over and self.saved_id is None:
            try:
                self.saved_id = self.save_game()
            except OSError as e:
                print(f"save game failed: {e}")
        return ev

    # ---- 复盘：已结束的对局存盘，回看时按布阵和着法重放，四家真实兵种都在
    def _ai_name(self, seat: int) -> str:
        if seat == self.human:
            return "人类"
        kind = self.ai.get(seat, "rule_soft")
        if kind == "neural" and self.model_path.exists():
            return f"神经网络（{self.model_path.parent.name}）"
        return next((k["name"] for k in self.ai_kinds() if k["id"] == kind), kind)

    def save_game(self) -> str:
        g = self.game
        GAMES.mkdir(parents=True, exist_ok=True)
        gid, k = time.strftime("%Y%m%d-%H%M%S"), 1
        while (GAMES / f"{gid}.json").exists():
            gid, k = f"{gid.split('-')[0]}-{gid.split('-')[1]}-{k}", k + 1
        rec = {"version": 1, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "human": self.human,
               "names": [self._ai_name(s) for s in range(4)], "origin": self.origin,
               "moves": [[e["from"], e["to"]] for e in self.events], "winner": g.winner,
               "team": [g.team_of(s) for s in range(4)], "end_reason": END_REASONS.get(g.end_reason, ""),
               "plies": len(self.events)}
        (GAMES / f"{gid}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        return gid

    @staticmethod
    def list_games(limit: int = 200) -> dict:
        items = []
        for p in sorted(GAMES.glob("*.json"), reverse=True)[:limit] if GAMES.exists() else []:
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            items.append({"id": p.stem, "saved": r.get("saved", ""), "human": r["human"], "winner": r["winner"],
                          "team": r["team"], "plies": r["plies"], "end_reason": r.get("end_reason", ""),
                          "origin": r["origin"].get("kind")})
        return {"items": items}

    def replay(self, gid: str) -> dict:
        """复盘一局：frames[i] 是走完 i 步后的完整局面 [[节点, 兵种, 座位], ...]，events[i] 是第 i+1 步。"""
        if not GAME_ID.match(gid or "") or not (GAMES / f"{gid}.json").exists():
            return {"error": "没有这局棋"}
        r = json.loads((GAMES / f"{gid}.json").read_text(encoding="utf-8"))
        g = self._replay(r["origin"], [])
        if g is None:
            return {"error": "这局的初始局面无法复原"}

        def frame():
            types, owners = g.board()
            occ = [int(n) for n in np.flatnonzero(owners >= 0)]
            mat = [0.0] * 4
            for n in occ:
                mat[int(owners[n])] += VALUE[types[n]]
            return {"p": [[n, int(types[n]), int(owners[n])] for n in occ], "mat": [round(float(m), 1) for m in mat],
                    "alive": [g.seat_alive(s) for s in range(4)], "flag": [g.flag_revealed(s) for s in range(4)],
                    "seat": g.current_seat}

        frames, events = [frame()], []
        for frm, to in r["moves"]:
            types, owners = g.board()
            seat, path = g.current_seat, self._path(frm, to, g)
            oc = g.step(frm, to)
            if oc < 0:
                return {"error": f"第 {len(events) + 1} 步重放失败"}
            events.append({"ply": len(events) + 1, "seat": seat, "from": frm, "to": to, "path": path,
                           "outcome": OUTCOME_NAMES[oc], "mt": int(types[frm]),
                           "tt": int(types[to]) if owners[to] >= 0 else None,
                           "ts": int(owners[to]) if owners[to] >= 0 else None})
            frames.append(frame())
        return {"id": gid, **{k: r[k] for k in ("saved", "human", "names", "winner", "team", "end_reason")},
                "frames": frames, "events": events}

    # ---- 分析帧：只用你这一方看得到的信息（自己的子、公开的位置、引擎推出的候选兵种），不含上帝视角
    def _view(self) -> list[tuple[int, int, int, int]]:
        """[(节点, 座位, 已知兵种或 -1, 候选掩码)]"""
        types, owners = self.game.board()
        masks = self.game.belief(self.human)
        out = []
        for n in map(int, np.flatnonzero(owners >= 0)):
            o = int(owners[n])
            m = 1 << int(types[n]) if o == self.human else int(masks[n])
            out.append((n, o, m.bit_length() - 1 if m and not m & (m - 1) else -1, m))
        return out

    @staticmethod
    def _summary(view) -> tuple[dict, dict]:
        """每家：子数、暗子数、估计子力、未知度（暗子身份熵之和，bit）。
        暗子的兵种分布 = 候选掩码，再做迭代比例拟合，使每种兵种的期望个数等于
        “编制数 − 已确认在场数”按暗子个数等比例缩放（开局时就是编制本身，估计子力为满编 88）。
        同时返回每颗暗子的这个“规则分布”（没有模型时显示它，终局时和神经信念比准确率）。"""
        mat, ent, pieces, hidden = [0.0] * 4, [0.0] * 4, [0] * 4, [0] * 4
        rule: dict[int, np.ndarray] = {}
        for o in range(4):
            mine = [(t, m) for _, s, t, m in view if s == o]
            pieces[o] = len(mine)
            known = np.zeros(12, np.int64)
            for t, _ in mine:
                if t >= 0:
                    known[t] += 1
                    mat[o] += VALUE[t]
            hid_nodes = [n for n, s, t, _ in view if s == o and t < 0]
            cand = np.array([(m >> BITS) & 1 for t, m in mine if t < 0], np.float64).reshape(-1, 12)
            hidden[o] = len(cand)
            if not len(cand):
                continue
            rem = np.maximum(QUOTA - known, 0).astype(np.float64)
            target = rem * len(cand) / max(rem.sum(), 1.0)  # 阵亡的暗子身份不明，按剩余编制等比例摊
            P = cand * np.maximum(rem, 1e-3)
            P /= P.sum(1, keepdims=True)
            for _ in range(60):
                P *= target / np.maximum(P.sum(0), 1e-9)
                P = np.where(cand > 0, np.maximum(P, 1e-12), 0.0)
                P /= P.sum(1, keepdims=True)
            rule.update(zip(hid_nodes, P))
            mat[o] += float((P @ VALUE).sum())
            nz = P[P > 1e-12]
            ent[o] += float(-(nz * np.log2(nz)).sum())
        return {"mat": [round(float(x), 1) for x in mat], "ent": [round(float(x), 1) for x in ent],
                "pieces": pieces, "hidden": hidden}, rule

    def _push_frame(self) -> dict:
        g = self.game
        view = self._view()
        summ, rule = self._summary(view)
        f = {"ply": g.ply, "view": view, "sum": summ, "rule": rule, "seat": g.current_seat}
        a = self.analyst()
        if a is not None:
            r = a.evaluate(g, self.human)
            f["wdl"] = r["wdl"]
            f["probs"] = {n: r["probs"][n] for n in rule}
            moves = r.get("moves")
            if moves is None and not g.over:
                moves = a.predict(g, self.human, r["probs"], np.random.default_rng(g.ply))
            f["moves"] = moves or []
        # 兵力估计：每家每种子大概还剩几个 = 已确认在场的个数 + 暗子身份分布（神经网络，没有模型时用规则分布）之和
        exp = np.zeros((4, 12))
        for n, o, t, _ in view:
            if t >= 0:
                exp[o, t] += 1
            else:
                exp[o] += f.get("probs", rule)[n]
        f["exp"] = np.round(np.minimum(exp, QUOTA), 2).tolist()  # 神经分布不受编制约束，显示时不超过编制数
        types, _ = g.board()
        f["truth"] = {n: int(types[n]) for n in rule}  # 私有：只在终局后用来算读子准确率，从不发给网页
        self.frames.append(f)
        return f

    @staticmethod
    def _probs(f: dict, n: int) -> list[float] | None:
        p = f.get("probs", f["rule"]).get(n)
        return None if p is None else [round(float(x), 3) for x in p]

    def stats(self) -> dict:
        """走法预测命中、你与 AI 首选的一致度；终局后再给读子准确率（神经信念 vs 规则分布的首选兵种）。"""
        pred = [e["rank"] for e in self.events if e.get("rank_kind") == "predict"]
        by_seat = {s: [e["rank"] for e in self.events if e.get("rank_kind") == "predict" and e["seat"] == s] for s in range(4)}
        sugg = [e["rank"] for e in self.events if e.get("rank_kind") == "suggest"]
        out = {"pred_n": len(pred), "pred_top1": sum(r == 1 for r in pred), "pred_top3": sum(r > 0 for r in pred),
               "agree_n": len(sugg), "agree_top1": sum(r == 1 for r in sugg), "agree_top3": sum(r > 0 for r in sugg),
               "pred_seat": [[len(r), sum(x == 1 for x in r), sum(x > 0 for x in r)] for r in by_seat.values()]}
        if self.game is not None and self.game.over:
            n = nm = hit = rule_hit = 0
            for f in self.frames:
                for node, t in f["truth"].items():
                    n += 1
                    rule_hit += int(np.argmax(f["rule"][node]) == t)
                    if "probs" in f:
                        nm += 1
                        hit += int(np.argmax(f["probs"][node]) == t)
            out["read_rule"] = round(rule_hit / n, 3) if n else None
            out["read_model"] = round(hit / nm, 3) if nm else None
        return out

    def timeline(self, since: int = 0) -> dict:
        """曲线数据：第 since 帧起的每帧汇总（state() 只带最后几帧，网页按 start 拼接）。"""
        return {"start": since, "items": [{"ply": f["ply"], "ev": f.get("ev"), "hit": f.get("hit"),
                                           "wdl": f.get("wdl"), **f["sum"]} for f in self.frames[since:]]}

    def frame(self, i: int) -> dict:
        """回看：第 i 帧的局面（与 state() 里 pieces 同样的格式）和到那时为止的最近几步。"""
        if not 0 <= i < len(self.frames):
            return {"error": "没有这一步"}
        f = self.frames[i]
        pieces = []
        for n, o, t, m in f["view"]:
            p = {"node": n, "owner": o}
            if o != self.human:
                p["cand"] = cand_list(m)
            if t >= 0:
                p["type"] = t
            elif (pr := self._probs(f, n)) is not None:
                p["prob"] = pr
            pieces.append(p)
        return {"i": i, "ply": f["ply"], "pieces": pieces, "events": self.events[max(0, i - 4):i],
                "moves": f.get("moves"), "move_seat": f["seat"], "wdl": f.get("wdl"), "exp": f["exp"]}

    def human_move(self, frm: int, to: int) -> dict:
        g = self.game
        if g is None or g.over or g.current_seat != self.human:
            return {"error": "现在不是你走"}
        path = self._path(frm, to)
        oc = g.step(frm, to)
        if oc < 0:
            return {"error": "走法不合法"}
        self._record(self.human, frm, to, oc, path)
        return self.state()

    def ai_step(self) -> dict:
        g = self.game
        if g is None or g.over or g.current_seat == self.human:
            return self.state()
        seat = g.current_seat
        frm, to = self.ai_move(seat)
        path = self._path(frm, to)
        oc = g.step(frm, to)
        if oc < 0:  # 理论上不会发生；保底用规则 AI
            frm, to = g.agent_act(AGENT_RULE, 0.0, self.rng)
            path = self._path(frm, to)
            oc = g.step(frm, to)
        self._record(seat, frm, to, oc, path)
        return self.state()

    def state(self) -> dict:
        g = self.game
        if g is None:
            return {"started": False}
        types, owners = g.board()
        masks = g.belief(self.human)
        over = g.over
        last = self.frames[-1]
        pieces = []
        for n in range(129):
            o = int(owners[n])
            if o < 0:
                continue
            p = {"node": n, "owner": o}
            if o == self.human or over:
                p["type"] = int(types[n])
            else:
                m = int(masks[n])
                cand = [t for t in range(12) if m >> t & 1]
                p["cand"] = cand
                if len(cand) == 1:
                    p["type"] = cand[0]
                else:
                    if (pr := self._probs(last, n)) is not None:
                        p["prob"] = pr
                    if self.god:
                        p["true"] = int(types[n])
            pieces.append(p)
        legal: dict[int, list[int]] = {}
        if not over and g.current_seat == self.human:
            for f, t in g.legal_moves():
                legal.setdefault(int(f), []).append(int(t))
        return {
            "started": True,
            "human": self.human,
            "current": g.current_seat,
            "over": over,
            "winner": g.winner,
            "end_reason": END_REASONS.get(g.end_reason, ""),
            "ply": g.ply,
            "alive": [g.seat_alive(s) for s in range(4)],
            "flag_revealed": [g.flag_revealed(s) for s in range(4)],
            "team": [g.team_of(s) for s in range(4)],
            "pieces": pieces,
            "legal": legal,
            "events": self.events[-80:],
            "timeline_tail": self.timeline(max(0, len(self.frames) - 3)),
            "moves": last.get("moves"),
            "move_seat": last["seat"],
            "wdl": last.get("wdl"),
            "exp": last["exp"],
            "model": "probs" in last,
            "stats": self.stats(),
            "ai": self.ai,
            "god": self.god,
            "origin": self.origin.get("kind"),
            "saved_id": self.saved_id,
        }

    def board_json(self) -> dict:
        b = self.info
        n = len(b.xy)
        edges = [[i, j, int(b.edge_kind[i, j])] for i in range(n) for j in range(i + 1, n) if b.edge_kind[i, j]]
        nodes = [{"id": i, "x": int(b.xy[i, 0]), "y": int(b.xy[i, 1]), "rail": bool(b.is_rail[i]),
                  "camp": bool(b.is_camp[i]), "base": bool(b.is_base[i]), "seat": int(b.seat[i]),
                  "row": int(b.row[i]), "col": int(b.col[i])} for i in range(n)]
        return {"nodes": nodes, "edges": edges, "types": TYPE_NAMES, "ai_kinds": self.ai_kinds()}


def cand_list(mask: int) -> list[int]:
    return [t for t in range(12) if mask >> t & 1]


def make_handler(sess: Session):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 静默
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            with sess.lock:
                if path == "/api/board":
                    return self._json(sess.board_json())
                if path == "/api/state":
                    return self._json(sess.state())
                if path == "/api/snapshot":
                    return self._json(sess.snapshot())
                if path == "/api/timeline":
                    return self._json(sess.timeline())
                if path == "/api/frame":
                    q = parse_qs(urlsplit(self.path).query)
                    return self._json(sess.frame(int(q.get("i", ["-1"])[0])))
                if path == "/api/games":
                    return self._json(sess.list_games())
                if path == "/api/replay":
                    q = parse_qs(urlsplit(self.path).query)
                    return self._json(sess.replay(q.get("id", [""])[0]))
                if path == "/api/setup":
                    return self._json({"setup": generate_setup(SETUP, sess.rng).tolist()})
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"error": "bad json"}, 400)
            path = self.path.split("?")[0]
            with sess.lock:
                if path == "/api/validate":
                    return self._json({"ok": valid_setup(np.array(body.get("setup", []), np.int8))
                                       if len(body.get("setup", [])) == 30 else False})
                if path == "/api/new":
                    return self._json(sess.new_game(int(body.get("human", 0)), body.get("setups", {}),
                                                    body.get("ai", {})))
                if path == "/api/load":
                    return self._json(sess.load_position(body.get("types", []), body.get("owners", []),
                                                         int(body.get("current", 0)), int(body.get("human", 0)),
                                                         body.get("ai", {})))
                if path == "/api/undo":
                    return self._json(sess.undo())
                if path == "/api/hint":
                    return self._json(sess.hint())
                if path == "/api/god":
                    sess.god = bool(body.get("on"))
                    return self._json(sess.state())
                if path == "/api/move":
                    return self._json(sess.human_move(int(body["from"]), int(body["to"])))
                if path == "/api/ai":
                    return self._json(sess.ai_step())
            self._send(404, b"not found", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--focus-bonus", type=float, default=FOCUS_BONUS, help="集火偏好的加分，0 关闭")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(Session(a.model, a.focus_bonus)))
    print(f"四国军旗人机对弈: http://{a.host}:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
