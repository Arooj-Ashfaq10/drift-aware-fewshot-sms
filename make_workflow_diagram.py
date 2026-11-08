"""Workflow diagram of the proposed drift-aware few-shot framework (figure for the paper)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 22); ax.set_ylim(0, 11); ax.axis("off")

def box(x, y, w, h, text, fc="#E8EEF7", ec="#2F4B7C", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.25", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")

def arrow(p, q, text=None, style="-|>", color="#333", ls="-", rad=0.0, off=(0, 0.25)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=1.3, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, fontsize=7.5, ha="center", color=color)

box(0.3, 4.2, 3.6, 2.6, "Time-ordered SMS stream\n(periods 0 ... T)\nmessage, label, period", "#FFF4D6", "#B8860B", bold=True)
box(5.0, 7.6, 4.2, 2.2, "Labelled pool\nperiod t-1 window (recent)\nor all periods < t (random)")
box(5.0, 4.2, 4.2, 2.6, "Example selector\nclass-balanced k examples\nRecent | Random | Triggered", bold=True)
box(10.3, 4.2, 3.7, 2.6, "Prompt builder\nsystem prompt (task)\n+ k labelled examples\n(user turn = incoming SMS)", bold=True, fs=8.5)
box(15.1, 4.2, 3.4, 2.6, "Frozen LLM scorer\ns = log P(Spam) - log P(Ham)\nSpam if s > 0", "#E4F4E8", "#2E7D32", bold=True, fs=8.5)
box(19.2, 4.2, 2.5, 2.6, "Predictions\non period t\n+ metrics", "#FFF4D6", "#B8860B", bold=True)
box(15.4, 0.4, 6.3, 2.4, "Drift monitor (label-free)\nd_t = 1 - cos(centroid of predicted-spam\nat reference, centroid at period t)", "#FBE9E7", "#C62828", bold=True, fs=8.5)
box(8.2, 0.4, 6.0, 2.4, "Trigger: d_t > tau\ntau = mean + z*std of split-half noise\nlabel k newest messages, refresh", "#FBE9E7", "#C62828", bold=True, fs=8.5)

arrow((3.9, 5.5), (5.0, 5.5))
arrow((2.1, 6.8), (5.0, 8.7), "labels of past periods", rad=-0.15, off=(-0.4, 0.45))
arrow((7.1, 7.6), (7.1, 6.8))
arrow((9.2, 5.5), (10.3, 5.5), "k examples", off=(0, 0.35))
arrow((14.0, 5.5), (15.1, 5.5)); arrow((18.5, 5.5), (19.2, 5.5))
arrow((20.4, 4.2), (20.4, 2.8), "predicted-spam texts", color="#C62828", off=(-1.9, 0.0))
arrow((15.4, 1.6), (14.2, 1.6), color="#C62828")
arrow((8.6, 2.8), (8.6, 4.2), "refresh for next period", color="#C62828", ls="--", off=(1.7, -0.1))
ax.text(11, 10.6, "Drift-aware few-shot adaptation of an LLM for evolving SMS spam", ha="center", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig("figures/workflow.png", dpi=220); plt.savefig("figures/workflow.pdf")
