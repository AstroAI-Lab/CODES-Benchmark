"""Draw the CODES workflow schematic used in the JOSS paper (figures/workflow.png)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREEN, GREY = "#2a6fdb", "#e8833a", "#2e9d6a", "#6b7280"


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=color + "1f", ec=color, lw=1.6,
        )
    )
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.52, "\n".join(lines), ha="center", va="top",
            fontsize=7.6, color="#1f2937", linespacing=1.35)


def arrow(ax, p, q, color=GREY, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=1.4, linestyle=ls))


fig, ax = plt.subplots(figsize=(10.5, 3.7))
ax.set_xlim(0, 10.45)
ax.set_ylim(0, 3.75)
ax.axis("off")

w, h, y = 1.85, 1.6, 1.95
xs = [0.05 + i * 2.1 for i in range(5)]
box(ax, xs[0], y, w, h, "Datasets", ["HDF5 train/val/test", "16 curated ODE datasets", "optional parameters", "log / min-max scaling"], GREY)
box(ax, xs[1], y, w, h, "Tune", ["Optuna studies", "SQLite or PostgreSQL", "pruning, multi-GPU", "accuracy + runtime"], ORANGE)
box(ax, xs[2], y, w, h, "Train", ["same seeds and data", "parallel workers", "checkpointing", "per-study model sets"], BLUE)
box(ax, xs[3], y, w, h, "Finetune", ["(optional)", "autoregressive", "rollout training", "Optuna search"], BLUE)
box(ax, xs[4], y, w, h, "Evaluate", ["accuracy, timing,", "compute, gradients", "interp. / extrap. / sparse", "batch scaling, UQ, rollout"], GREEN)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + w + 0.02, y + h / 2), (b - 0.02, y + h / 2))

# surrogate registry feeding tune/train/finetune/evaluate
rx, ry, rw, rh = xs[1], 0.1, xs[4] + w - xs[1], 1.05
box(ax, rx, ry, rw, rh, "Surrogate registry (AbstractSurrogateModel)",
    ["MultiONet (DeepONet)  |  FullyConnected  |  LatentNeuralODE  |  LatentPoly  |  residual variants",
     "shared prepare_data / fit / predict / validate / save interface; new models register by name"],
    "#7c3aed")
for x0 in xs[1:]:
    arrow(ax, (x0 + w / 2, ry + rh + 0.02), (x0 + w / 2, y - 0.03), color="#7c3aed", ls="--")
ax.text(xs[0] + w / 2, 0.62, "one YAML config\ndrives every stage", ha="center", va="center",
        fontsize=8.5, style="italic", color=GREY)

out = Path(__file__).with_name("workflow.png")
fig.savefig(out, dpi=250, bbox_inches="tight")
print(out)
