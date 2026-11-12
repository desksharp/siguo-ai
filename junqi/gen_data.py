"""自对弈数据生成：按“对局配方”混合产出 .jqr 分片，并写 manifest.json 记录来源。

示例：
  python3 -m junqi.gen_data --rules duel --games 200000 --out data/v2_rule/duel \
      --mix "rule:0.1/rule:0.1=0.5,rule:0.3/rule:0.3=0.3,rule:0.1/random=0.1,random/rule:0.1=0.1"

配方格式：<队伍0智能体>/<队伍1智能体>=<比例>，智能体为 random 或 rule:<温度>。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .engine import AGENT_RANDOM, AGENT_RULE, SETUP_HEURISTIC, SETUP_RANDOM, load_rules, selfplay


def parse_agent(s: str) -> tuple[int, float]:
    s = s.strip()
    if s == "random":
        return AGENT_RANDOM, 0.0
    if s.startswith("rule"):
        return AGENT_RULE, float(s.split(":")[1]) if ":" in s else 0.1
    raise ValueError(f"unknown agent {s}")


def parse_mix(mix: str) -> list[tuple[str, str, float]]:
    out = []
    for part in mix.split(","):
        pair, w = part.split("=")
        a, b = pair.split("/")
        out.append((a, b, float(w)))
    tot = sum(w for *_, w in out)
    return [(a, b, w / tot) for a, b, w in out]


def generate(rules: str, games: int, out: Path, mix: str, seed: int = 0, shard_games: int = 20000,
             threads: int = 0, random_setup_frac: float = 0.2) -> dict:
    cfg = load_rules(rules)
    out.mkdir(parents=True, exist_ok=True)
    duel = cfg.mode == 1
    teams = ([0], [2]) if duel else ([0, 2], [1, 3])
    manifest = {"rules": rules, "config": cfg.to_dict(), "mix": mix, "seed": seed, "shards": [], "created": time.time()}
    shard, done = 0, 0
    t0 = time.time()
    while done < games:
        n = min(shard_games, games - done)
        path = out / f"shard_{shard:04d}.jqr"
        path.unlink(missing_ok=True)
        totals = {"team0": 0, "team1": 0, "draws": 0, "plies": 0}
        for ri, (a, b, w) in enumerate(parse_mix(mix)):
            k = max(1, round(n * w))
            for setup_kind, frac in ((SETUP_HEURISTIC, 1 - random_setup_frac), (SETUP_RANDOM, random_setup_frac)):
                kk = round(k * frac)
                if kk <= 0:
                    continue
                kinds, params = [0] * 4, [0.0] * 4
                for team, ag in ((0, parse_agent(a)), (1, parse_agent(b))):
                    for seat in teams[team]:
                        kinds[seat], params[seat] = ag
                st = selfplay(cfg, kk, seed=seed + shard * 1_000_003 + ri * 7919 + setup_kind, threads=threads,
                              path=path, append=True, agent_kind=kinds, agent_param=params,
                              setup_kind=[setup_kind] * 4)
                for key in totals:
                    totals[key] += st[key]
        manifest["shards"].append({"file": path.name, **totals})
        done += n
        shard += 1
        print(f"  {path.name}: {totals}  ({done}/{games}, {time.time() - t0:.0f}s)", flush=True)
    manifest["games"] = sum(s["team0"] + s["team1"] + s["draws"] for s in manifest["shards"])
    manifest["plies"] = sum(s["plies"] for s in manifest["shards"])
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", default="duel")
    ap.add_argument("--games", type=int, default=100000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mix", default="rule:0.1/rule:0.1=0.5,rule:0.3/rule:0.3=0.3,rule:0.1/random=0.1,random/rule:0.1=0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-games", type=int, default=20000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--random-setup-frac", type=float, default=0.2)
    a = ap.parse_args()
    m = generate(a.rules, a.games, a.out, a.mix, a.seed, a.shard_games, a.threads, a.random_setup_frac)
    print(f"done: {m['games']} games, {m['plies']} plies → {a.out}")


if __name__ == "__main__":
    main()
