"""Result figures from results/<mode>/ (run after run_drift.py)."""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results/simulated"); a = ap.parse_args()
pp = pd.read_csv(os.path.join(a.dir, "per_period.csv")); sm = pd.read_csv(os.path.join(a.dir, "summary.csv"))
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for pol, g in pp.groupby("policy"):
    m = g.groupby("period")["f1"].mean(); ax.plot(m.index, m.values, marker="o", ms=3, label=pol)
ax.set_xlabel("Period (0 = stable, >=1 = drifted)"); ax.set_ylabel("F1 (mean over seeds)"); ax.grid(alpha=.3)
ax.legend(fontsize=6, ncol=2); plt.tight_layout(); plt.savefig("figures/f1_by_period.png", dpi=200); plt.close()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.errorbar(sm.labels_used_mean, sm.f1_mean, yerr=sm.f1_std.fillna(0), fmt="o", capsize=3)
for _, r in sm.iterrows():
    ax.annotate(r.policy, (r.labels_used_mean, r.f1_mean), fontsize=6, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Newly labelled examples used"); ax.set_ylabel("F1 on drifted periods"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figures/cost_vs_f1.png", dpi=200); plt.close()
dm = os.path.join(a.dir, "drift_monitor.csv")
if os.path.exists(dm) and os.path.getsize(dm) > 5:
    d = pd.read_csv(dm)
    if len(d):
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for s, g in d.groupby("seed"):
            ax.plot(g.period, g.drift, marker="o", label=f"d_t seed {s}"); ax.axhline(g.tau.iloc[0], ls="--", lw=.8)
        ax.set_xlabel("Period"); ax.set_ylabel("Drift distance d_t"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("figures/drift_monitor.png", dpi=200); plt.close()
print("figures saved in figures/")
