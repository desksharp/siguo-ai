"""风格体检：胜率之外，看 AI 下棋像不像人（四家同一个智能体自对弈）。

  python3 -m junqi.style --agent runs/rl_four_v3/model.pt --temp 0.5 --games 300
  python3 -m junqi.style --agent rule          # 规则 AI 作参照；exploit = 专打暴露后排的规则 AI

指标（“后两排”指自家营区第 5、6 行，不含大本营；“压境”指有敌子进到自家营区第 3 行及以后）：
  hq_waste            另一个大本营放了工兵、营长以上大子或炸弹（每家的比例，只取决于布阵）
  pressed_back        压境时，这一步走的是后两排子的比例
  calm_back           没有压境时，同上
  pressed_back_move   压境时，后两排子不吃子、单纯挪动的比例（白白告诉对方“这颗不是雷”）
  reveal_per_seat     每家每局“暴露”次数：后两排子走到空位，而对手此前还认为它可能是地雷
  focus_same          队友刚攻击过某家敌人（8 步内），我攻击同一家的比例；两家敌人随机选约 0.5
  undo_move           把刚走过去的子又走回原位的比例（等待着，太高就是在拖）
  draw_rate           和棋局的比例（400 步没有碰子判和）
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .engine import AGENT_EXPLOIT, AGENT_RULE, SETUP_HEURISTIC, Game, board_info, load_rules

FLAG, MINE, PLATOON, COMPANY = 0, 1, 4, 5
MINE_BIT = 1 << MINE


def style_report(cfg, agent, n_games: int = 200, seed: int = 7, batch: int = 100, setup_kind: int = SETUP_HEURISTIC) -> dict:
    B = board_info(cfg.arc_mode, cfg.center_middle_rail, cfg.camp_corner_straight)
    seat_of, row = B.seat.astype(np.int64), B.row.astype(np.int64)
    base = B.is_base.astype(bool)
    back = np.stack([(seat_of == s) & (row >= 4) & ~base for s in range(4)])  # [4,129]
    front_zone = np.stack([(seat_of == s) & (row >= 2) for s in range(4)])
    c = dict(seats=0, hq_waste=0, pressed=0, pressed_back=0, pressed_back_move=0, calm=0, calm_back=0,
             reveal=0, attacks=0, focus_pairs=0, focus_same=0, plies=0, games=0, moves=0, undo=0, draws=0)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_games:
        k = min(batch, n_games - done)
        games = [Game(cfg).reset(int(rng.integers(2**62)), setup_kinds=[setup_kind] * 4) for _ in range(k)]
        team = np.array([games[0].team_of(s) for s in range(4)] + [-1])  # team[-1]：空位
        last_att = [dict() for _ in range(k)]
        last_move = [dict() for _ in range(k)]
        for g in games:
            t, _ = g.board()
            for s in range(4):
                hq = t[base & (seat_of == s)]
                c["hq_waste"] += int(np.sum(~np.isin(hq, (FLAG, MINE, PLATOON, COMPANY))))
            c["seats"] += 4
        live = list(range(k))
        while live:
            seats = [games[i].current_seat for i in live]
            boards = [games[i].board() for i in live]
            moves = agent.act_batch([games[i] for i in live], seats)
            for i, s, (t, o), (f, to) in zip(live, seats, boards, moves):
                g = games[i]
                my = team[s]
                enemy = (o >= 0) & (team[o] != my)
                pressed = bool(np.any(enemy & front_zone[s]))
                is_back = bool(back[s, f])
                c["moves"] += 1
                c["undo"] += last_move[i].get(s) == (to, f)
                last_move[i][s] = (f, to)
                attack = bool(enemy[to])
                if pressed:
                    c["pressed"] += 1
                    c["pressed_back"] += is_back
                    c["pressed_back_move"] += is_back and not attack
                else:
                    c["calm"] += 1
                    c["calm_back"] += is_back
                if is_back and o[to] < 0:
                    foe = next(x for x in range(4) if team[x] != my and g.seat_alive(x))
                    c["reveal"] += bool(g.belief(foe)[f] & MINE_BIT)
                if attack:
                    ts = int(o[to])
                    c["attacks"] += 1
                    mate = last_att[i].get((s + 2) % 4)
                    if mate is not None and g.ply - mate[0] <= 8:
                        c["focus_pairs"] += 1
                        c["focus_same"] += mate[1] == ts
                    last_att[i][s] = (g.ply, ts)
                if g.step(f, to) < 0:
                    raise RuntimeError(f"illegal move {f}->{to}")
            live = [i for i in live if not games[i].over]
        for g in games:
            c["plies"] += g.ply
            c["games"] += 1
            c["draws"] += g.winner < 0
        done += k
    r = lambda a, b: round(c[a] / max(c[b], 1), 4)  # noqa: E731
    return dict(
        games=c["games"],
        hq_waste=r("hq_waste", "seats"),
        pressed_back=r("pressed_back", "pressed"),
        calm_back=r("calm_back", "calm"),
        pressed_back_move=r("pressed_back_move", "pressed"),
        reveal_per_seat=r("reveal", "seats"),
        focus_same=r("focus_same", "focus_pairs"),
        avg_plies=r("plies", "games"),
        undo_move=r("undo", "moves"),
        draw_rate=r("draws", "games"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="rule", help="rule / exploit / 模型路径")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--rules", default="siguo_standard")
    ap.add_argument("--setup", type=int, default=SETUP_HEURISTIC, help="1 现行启发式，2 旧版（对照）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = load_rules(a.rules)
    if a.agent in ("rule", "exploit"):
        from .arena import CppAgent
        agent = CppAgent(AGENT_RULE if a.agent == "rule" else AGENT_EXPLOIT, a.temp)
    else:
        import torch

        from .agents.neural import NeuralAgent
        from .model import load
        dev = a.device if torch.cuda.is_available() else "cpu"
        agent = NeuralAgent(load(a.agent, dev), dev, temperature=a.temp, seed=a.seed)
    print(json.dumps({"agent": a.agent, "temp": a.temp, **style_report(cfg, agent, a.games, a.seed, setup_kind=a.setup)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
