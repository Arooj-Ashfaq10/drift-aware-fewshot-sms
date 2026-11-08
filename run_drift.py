"""Main experiments: stable vs drifted evaluation, recent vs random example selection, refresh intervals,
and the proposed drift-triggered refresh (DTR)."""
import argparse, json, os
import numpy as np, pandas as pd
from src.common import build_system_prompt
from src.data import load_uci, build_simulated_stream, build_real_stream
from src.llm import CachedClassifier, make_backend
from src.metrics import metrics, bootstrap_ci, mcnemar
from src.strategies import pick_examples, window, history, calibrate_tau, drift_distance


def run_policy(clf, stream, kind, R, k, seed, z):
    P = int(stream.period.max()) + 1
    name = {"zs": "zs", "static": "static", "dtr": f"dtr_z{z:g}"}.get(kind) or f"{kind}_R{R}"
    ex = [] if kind == "zs" else pick_examples(window(stream, 0), k, seed)
    refreshes, pending, ref_texts, tau = 0, False, None, None
    rows, dlog = [], []
    for t in range(P):
        refreshed = False
        if t >= 2:
            if kind == "recent" and (t - 1) % R == 0:
                ex = pick_examples(window(stream, t - 1), k, seed * 1000 + t); refreshed = True
            elif kind == "random" and (t - 1) % R == 0:
                ex = pick_examples(history(stream, t), k, seed * 1000 + t); refreshed = True
            elif kind == "dtr" and pending:
                ex = pick_examples(window(stream, t - 1), k, seed * 1000 + t); refreshed = True
                pending, ref_texts = False, None
        refreshes += int(refreshed)
        ev = stream[(stream.period == t) & stream.evalset]
        sc = clf.score(build_system_prompt(ex), ev.text.tolist())
        pred = (sc > 0).astype(int)
        for mid, y, p, s in zip(ev.id, ev.label, pred, sc):
            rows.append(dict(policy=name, seed=seed, period=t, id=int(mid), y=int(y), pred=int(p), score=float(s), refreshed=refreshed))
        if kind == "dtr" and t >= 1:
            st = ev.text[pred == 1].tolist()
            if len(st) >= 10:
                if tau is None:
                    tau = calibrate_tau(st, z=z, seed=seed)
                if ref_texts is None:
                    ref_texts, d = st, 0.0
                else:
                    d = drift_distance(ref_texts, st)
                    pending = pending or d > tau
                dlog.append(dict(policy=name, seed=seed, period=t, drift=d, tau=tau, triggered=bool(pending)))
    labels = 0 if kind == "zs" else k * (1 + refreshes)
    return name, pd.DataFrame(rows), dict(policy=name, seed=seed, refreshes=refreshes, labels_used=labels), dlog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["simulated", "real"], default="simulated")
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--uci_path", default=None)
    ap.add_argument("--real_csv", default=None)
    ap.add_argument("--ads", default="exclude", choices=["exclude", "spam", "ham"])
    ap.add_argument("--period_months", type=int, default=3)
    ap.add_argument("--periods", type=int, default=6)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--intervals", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--z", type=float, nargs="+", default=[1.0, 2.0, 3.0], help="threshold multipliers for DTR variants")
    ap.add_argument("--max_eval", type=int, default=None, help="cap on evaluated messages per period")
    ap.add_argument("--reverse_axis", action="store_true", help="reverse the drift direction of the simulated stream")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = os.path.join(a.out, a.mode); os.makedirs(out, exist_ok=True)
    stream = (build_simulated_stream(load_uci(a.uci_path), n_periods=a.periods, seed=0, max_eval=a.max_eval, reverse=a.reverse_axis)
              if a.mode == "simulated" else
              build_real_stream(a.real_csv, a.period_months, a.ads, seed=0, max_eval=a.max_eval))
    stream.to_csv(os.path.join(out, "stream.csv"), index=False)
    desc = stream.groupby(["period", "label"]).size().unstack(fill_value=0).rename(columns={0: "ham", 1: "spam"})
    desc["evaluated"] = stream[stream.evalset].groupby("period").size()
    desc.to_csv(os.path.join(out, "dataset_stats.csv")); print(desc)

    clf = CachedClassifier(make_backend(a.backend, a.model, not a.no_4bit),
                           cache_path=os.path.join(out, f"cache_{a.backend}_{a.model.replace('/', '_')}.pkl"), batch_size=a.batch_size)
    specs = [("zs", 0, [0], 0.0), ("static", 0, a.seeds, 0.0)] + [(k_, R, a.seeds, 0.0) for R in a.intervals for k_ in ("recent", "random")] + [("dtr", 0, a.seeds, zz) for zz in a.z]
    preds, meta, dlogs = [], [], []
    for kind, R, seeds, zz in specs:
        for s in seeds:
            name, p, m, d = run_policy(clf, stream, kind, R, a.k, s, zz)
            preds.append(p); meta.append(m); dlogs += d
            print(f"done {name} seed={s} refreshes={m['refreshes']} cache_hits={clf.n_hits}", flush=True)
    P = pd.concat(preds); P.to_csv(os.path.join(out, "predictions.csv"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(out, "run_meta.csv"), index=False)
    pd.DataFrame(dlogs).to_csv(os.path.join(out, "drift_monitor.csv"), index=False)

    # ---- aggregate ----
    per_period = [dict(policy=pol, seed=s, period=t, **metrics(g.y, g.pred)) for (pol, s, t), g in P.groupby(["policy", "seed", "period"])]
    pd.DataFrame(per_period).to_csv(os.path.join(out, "per_period.csv"), index=False)
    M = pd.DataFrame(meta); rows = []
    for (pol, s), g in P.groupby(["policy", "seed"]):
        dr = g[g.period >= 1]; st = g[g.period == 0]
        rows.append(dict(policy=pol, seed=s, **{k_: v for k_, v in metrics(dr.y, dr.pred).items()}, stable_f1=metrics(st.y, st.pred)["f1"]))
    S = pd.DataFrame(rows).merge(M, on=["policy", "seed"])
    S.to_csv(os.path.join(out, "summary_by_seed.csv"), index=False)
    cols = ["acc", "precision", "recall", "f1", "fpr", "fnr", "stable_f1", "labels_used", "refreshes"]
    agg = S.groupby("policy")[cols].agg(["mean", "std"]); agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["n_seeds"] = S.groupby("policy").size(); agg["ms_per_msg_uncached"] = clf.ms_per_msg
    agg.to_csv(os.path.join(out, "summary.csv")); print(agg[["f1_mean", "f1_std", "stable_f1_mean", "labels_used_mean"]])

    # ---- significance (seed 0, pooled drifted periods) ----
    D = P[(P.period >= 1) & (P.seed == 0)]
    pairs = [(f"recent_R{R}", f"random_R{R}") for R in a.intervals] + [(f"recent_R{a.intervals[0]}", "static"), ("static", "zs")] + [(f"dtr_z{zz:g}", "static") for zz in a.z] + [(f"dtr_z{zz:g}", f"recent_R{a.intervals[0]}") for zz in a.z]
    tests = []
    for x, y_ in pairs:
        A, B = D[D.policy == x].set_index("id"), D[D.policy == y_].set_index("id")
        j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
        if len(j):
            b, c, pv = mcnemar(j.y_a, j.pred_a, j.pred_b)
            lo, hi = bootstrap_ci(A.y, A.pred)
            tests.append(dict(system_a=x, system_b=y_, only_a_correct=b, only_b_correct=c, mcnemar_p=pv, f1_a_ci_low=lo, f1_a_ci_high=hi))
    pd.DataFrame(tests).to_csv(os.path.join(out, "mcnemar_tests.csv"), index=False)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=2)
    print(pd.DataFrame(tests))


if __name__ == "__main__":
    main()
