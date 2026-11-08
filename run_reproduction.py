"""Reproduction of the base paper's (SpaLLM-Guard, Salman et al. 2025) zero-shot / few-shot protocol on UCI SMS."""
import argparse, json, os
import numpy as np, pandas as pd
from src.common import build_system_prompt
from src.data import load_uci
from src.llm import CachedClassifier, make_backend
from src.metrics import metrics
from src.strategies import pick_examples

# Values reported for Mistral-7B-Instruct-v0.3 in SpaLLM-Guard (Tables 1, 2, 3; the 5-example row is labelled
# 'Mixtral (7B)' in the paper). Their test set: 10,000 messages from a 67,018-message 'Super Dataset'.
PAPER = {
    "zero_shot":        dict(acc=.8040, f1=.8323, tpr=.9740, tnr=.6344, fpr=.3656, fnr=.0260),
    "fewshot_random10": dict(acc=.5235, f1=.6669, tpr=1.000, tnr=.0612, fpr=.9388, fnr=.0000),
    "fewshot_tuned5":   dict(acc=.7325, f1=.7880, tpr=.9900, tnr=.4727, fpr=.5273, fnr=.0100),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "mock"], default="hf")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--uci_path", default=None)
    ap.add_argument("--n_test", type=int, default=500, help="messages per class in the balanced test set")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/reproduction")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    df = load_uci(a.uci_path)
    rng = np.random.default_rng(123)
    te = pd.concat([df[df.label == y].sample(min(a.n_test, (df.label == y).sum() // 2), random_state=123) for y in (0, 1)])
    pool = df[~df.id.isin(te.id)]
    print(f"UCI after de-duplication: {len(df)} (spam={int(df.label.sum())}); test={len(te)}")
    clf = CachedClassifier(make_backend(a.backend, a.model, not a.no_4bit),
                           cache_path=os.path.join(a.out, f"cache_{a.backend}_{a.model.replace('/', '_')}.pkl"), batch_size=a.batch_size)
    configs = {  # name: (n_spam, n_ham) ; paper: 10 random spam (5 fraud + 5 promo), and 5 tuned examples (2+2 spam, 1 ham)
        "zero_shot": None, "fewshot_random10": (10, 0), "fewshot_tuned5": (4, 1), "fewshot_balanced10": (5, 5)}
    rows = []
    for name, cnt in configs.items():
        for s in ([0] if cnt is None else a.seeds):
            ex = [] if cnt is None else pick_examples(pool, n_spam=cnt[0], n_ham=cnt[1], seed=s)
            sc = clf.score(build_system_prompt(ex), te.text.tolist())
            m = metrics(te.label, (sc > 0).astype(int))
            rows.append(dict(config=name, seed=s, **m)); print(name, s, {k: round(v, 4) for k, v in m.items() if k != 'n'}, flush=True)
    R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out, "reproduction_by_seed.csv"), index=False)
    agg = R.groupby("config")[["acc", "f1", "recall", "tnr", "fpr", "fnr"]].agg(["mean", "std"]); agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.to_csv(os.path.join(a.out, "reproduction_summary.csv")); print(agg.round(4))
    json.dump(PAPER, open(os.path.join(a.out, "paper_reported_mistral7b.json"), "w"), indent=2)
    json.dump(dict(vars(a), n_unique_msgs=len(df), n_test=len(te)), open(os.path.join(a.out, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
