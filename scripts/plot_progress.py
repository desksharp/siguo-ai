#!/usr/bin/env python3
"""从 runs/ 下的训练日志画 README 和技术报告里用的曲线图。

  python3 scripts/plot_progress.py --out docs/img

需要 matplotlib 和对应的训练目录（runs/bc_four_v5、rl_four_v1、rl_four_v6..v19）。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"
GENS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"

plt.rcParams.update({
    "font.family": ["Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", "sans-serif"],
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.unicode_minus": False,
})


def log(run: str) -> list[dict]:
    return [json.loads(x) for x in (RUNS / run / "log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def evals(run: str) -> list[tuple[int, dict]]:
    return [(r["iter"], r["eval"]) for r in log(run) if "eval" in r]


def smooth(xs: list[float], k: int) -> list[float]:
    return [sum(xs[max(0, i - k + 1):i + 1]) / len(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]


def ci95(w: int, l: int, d: int) -> float:
    n = w + l + d
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    return 1.96 * math.sqrt(var / n)


def half_line(ax, text: str, left: bool = False) -> None:
    ax.axhline(0.5, color=INK2, lw=1, ls="--", zorder=1)
    ax.annotate(text, (0 if left else 1, 0.5), xycoords=("axes fraction", "data"),
                xytext=(4 if left else -2, -4), textcoords="offset points",
                ha="left" if left else "right", va="top", color=INK2, fontsize=9)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name)


def plot_generations(out: Path) -> None:
    labels, score, err, undo, draw = [], [], [], [], []
    for g in GENS:
        s = json.loads((RUNS / f"rl_four_v{g}" / "summary.json").read_text(encoding="utf-8"))
        v = s["vs_bc"]
        labels.append(f"v{g}")
        score.append(v["score"])
        err.append(ci95(v["w"], v["l"], v["d"]))
        undo.append(100 * s["style"]["undo_move"])
        draw.append(100 * s["style"]["draw_rate"])
    x = list(range(len(GENS)))
    cut = GENS.index(16) - 0.5

    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    a.errorbar(x, score, yerr=err, color=BLUE, marker="o", ms=6, capsize=3, elinewidth=1.2, zorder=3)
    half_line(a, "与上一代持平", left=True)
    for i in (0, len(x) - 1):
        a.annotate(f"{score[i] + 1e-9:.3f}", (x[i], score[i]), xytext=(8, 0), textcoords="offset points",
                   va="center", color=INK, fontsize=9)
    a.set_ylim(0.45, 0.8)
    a.set_ylabel("得分")
    a.set_title("每代对上一代的得分（1000 局，误差线为 95% 置信区间）")

    b.plot(x, undo, color=BLUE, marker="o", ms=5, label="来回拖棋")
    b.plot(x, draw, color=ORANGE, marker="s", ms=5, label="和棋率")
    b.axvline(cut, color=INK2, lw=1, ls=":")
    b.annotate("v16 起加入 --undo-penalty", (cut, 46), xytext=(6, 0), textcoords="offset points",
               color=INK2, fontsize=9, va="top")
    b.annotate("来回拖棋", (x[GENS.index(14)], undo[GENS.index(14)]), xytext=(-6, 6),
               textcoords="offset points", ha="right", color=INK, fontsize=9)
    b.annotate("和棋率", (x[GENS.index(14)], draw[GENS.index(14)]), xytext=(-6, -12),
               textcoords="offset points", ha="right", color=INK, fontsize=9)
    b.set_ylim(0, 50)
    b.set_ylabel("%")
    b.set_title("棋风：来回拖棋比例与和棋率")
    b.legend(loc="upper left")
    b.set_xticks(x, labels)
    fig.tight_layout()
    save(fig, out, "generations.png")


def plot_rl_run(out: Path, run: str, prev: str) -> None:
    rows = [r for r in log(run) if "eval" not in r]
    ev = evals(run)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4))

    a = ax[0][0]
    it = [i for i, _ in ev]
    a.plot(it, [e["vs_rule"] for _, e in ev], color=BLUE, marker="o", ms=5, label="对规则 AI（老师）")
    a.plot(it, [e["vs_bc"] for _, e in ev], color=ORANGE, marker="s", ms=5, label=f"对起点模型 {prev}")
    half_line(a, "持平")
    a.set_ylim(0.4, 1.02)
    a.set_title("对战得分（每 20 轮评估 400 局）")
    a.legend(loc="center right")

    a = ax[0][1]
    rr = [(r["iter"], r["roll_rule"]) for r in rows if r.get("roll_rule_n")]
    a.plot([i for i, _ in rr], [v for _, v in rr], color=BLUE, lw=0.8, alpha=0.35, label="每轮")
    a.plot([i for i, _ in rr], smooth([v for _, v in rr], 10), color=BLUE, label="近 10 轮平均")
    a.set_ylim(0.94, 1.005)
    a.set_title("采样对局中对规则 AI 的得分")
    a.legend(loc="lower right")

    a = ax[1][0]
    a.plot([r["iter"] for r in rows], [r["entropy"] for r in rows], color=BLUE, lw=1)
    a.set_title("策略熵（越低走法越果断）")

    a = ax[1][1]
    a.plot([r["iter"] for r in rows], [r["kl_ref"] for r in rows], color=ORANGE, lw=1.5)
    a.set_ylim(bottom=0)
    a.set_title("偏离起点模型的程度（KL）")

    for a in ax[1]:
        a.set_xlabel("轮次")
    fig.suptitle(f"{run}：一代强化学习（200 轮）", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out, f"curves_{run}.png")


def plot_rl_from_bc(out: Path) -> None:
    ev = evals("rl_four_v1")
    fig, a = plt.subplots(figsize=(8, 3.6))
    it = [i for i, _ in ev]
    y = [e["vs_rule"] for _, e in ev]
    a.plot(it, y, color=BLUE, marker="o", ms=4)
    half_line(a, "老师水平")
    a.annotate(f"起点（行为克隆）{y[0]:.2f}", (it[0], y[0]), xytext=(8, -4), textcoords="offset points",
               va="top", color=INK, fontsize=9)
    best = max(range(len(y)), key=y.__getitem__)
    a.annotate(f"{y[best]:.2f}", (it[best], y[best]), xytext=(0, 8), textcoords="offset points",
               ha="center", color=INK, fontsize=9)
    a.set_ylim(0.4, 1.0)
    a.set_xlabel("轮次")
    a.set_ylabel("得分")
    a.set_title("强化学习超过老师：2.82M 模型对规则 AI 的得分（贪心走法，每次 400 局）")
    fig.tight_layout()
    save(fig, out, "rl_from_bc.png")


def plot_bc(out: Path, run: str) -> None:
    rows = log(run)
    tr = [r for r in rows if "eval" not in r and "policy_top1" in r]
    ev = [(r["step"], r["eval"]) for r in rows if "eval" in r]
    keys = [("policy_top1", "走法命中率"), ("belief_top1", "猜子命中率"), ("value_acc", "胜负预测准确率")]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, (k, title) in zip(ax, keys):
        a.plot([r["step"] for r in tr], smooth([r[k] for r in tr], 5), color=BLUE, lw=1.5, label="训练数据")
        a.plot([s for s, _ in ev], [e[k] for _, e in ev], color=ORANGE, marker="o", ms=6, lw=0,
               label="留出数据")
        a.set_title(title)
        a.set_xlabel("步数")
    ax[0].legend(loc="lower right")
    fig.suptitle(f"{run}：14.7M 模型行为克隆", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, out, f"curves_{run}.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/img")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_generations(out)
    plot_rl_run(out, "rl_four_v19", "v18")
    plot_rl_from_bc(out)
    plot_bc(out, "bc_four_v5")


if __name__ == "__main__":
    main()
