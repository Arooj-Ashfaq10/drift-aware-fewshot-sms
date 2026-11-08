"""Dataset loading and construction of time-ordered SMS streams.
Stream columns: id, text, label (1 spam / 0 ham), period (int), split ('pool'|'eval'), evalset (bool)."""
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def load_uci(path=None):
    """UCI SMS Spam Collection. path = tab-separated 'SMSSpamCollection' file, else Hugging Face ucirvine/sms_spam."""
    if path:
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE, encoding="latin-1")
        df["label"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    else:
        from datasets import load_dataset
        d = load_dataset("ucirvine/sms_spam", split="train").to_pandas()
        df = pd.DataFrame({"label": d["label"].astype(int), "text": d["sms"]})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df.text.str.len() > 0].drop_duplicates(subset="text").reset_index(drop=True)
    df["id"] = np.arange(len(df))
    return df[["id", "text", "label"]]


def _split_period0(df, seed, frac_pool=0.6):
    rng = np.random.default_rng(seed)
    df["split"] = "eval"
    for y in (0, 1):
        idx = df.index[(df.period == 0) & (df.label == y)].to_numpy()
        idx = rng.permutation(idx)
        df.loc[idx[: int(frac_pool * len(idx))], "split"] = "pool"
    return df


def _finalize(df, max_eval, seed):
    rng = np.random.default_rng(seed + 7)
    df["evalset"] = False
    df.loc[(df.period == 0) & (df.split == "eval"), "evalset"] = True
    for p in sorted(df.period.unique()):
        if p == 0:
            continue
        idx = df.index[df.period == p].to_numpy()
        if max_eval and len(idx) > max_eval:
            idx = rng.choice(idx, max_eval, replace=False)
        df.loc[idx, "evalset"] = True
    return df.reset_index(drop=True)


def build_simulated_stream(uci, n_periods=6, n_clusters=4, sigma=0.7, ham_ratio=3, seed=0, max_eval=None, method="axis", jitter=0.6, reverse=False):
    """Simulated concept drift on UCI.  method='axis' (default): spam messages are ordered along the dominant content axis
    of the spam TF-IDF space (first SVD component); a jittered rank assigns them to periods, so every period has the same
    number of spam messages and the spam *type* drifts gradually and monotonically (neighbouring periods overlap, period 0
    and period T-1 are far apart). Ham is sampled uniformly (stationary). method='cluster' is the earlier KMeans variant
    (kept for reference; its sequential depletion made the last period fall back to the first cluster)."""
    rng = np.random.default_rng(seed)
    spam = uci[uci.label == 1].copy(); ham = uci[uci.label == 0].copy()
    X = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(spam.text)
    n_sp = int(0.95 * len(spam) / n_periods); n_h = ham_ratio * n_sp
    ham_idx = rng.permutation(ham.index.to_numpy())[: n_h * n_periods]
    if method == "axis":
        comp = TruncatedSVD(1, random_state=seed).fit_transform(X)[:, 0]
        if reverse:
            comp = -comp          # drift in the opposite direction (P0 <-> P5 swapped)
        rank_q = (np.argsort(np.argsort(comp)) + 0.5) / len(comp)           # position on the drift axis in (0,1)
        noisy = rank_q * n_periods + rng.normal(0, jitter, len(comp))        # jitter -> overlapping neighbouring periods
        order = np.argsort(noisy)[: n_sp * n_periods]                        # equal-sized periods, ordered by noisy position
        spam = spam.iloc[order].copy()
        spam["period"] = np.repeat(np.arange(n_periods), n_sp)
        spam["cluster"] = np.round(rank_q[order] * 100).astype(int)          # 'cluster' column = axis percentile (0..100)
        parts = [spam]
        for p in range(n_periods):
            h = ham.loc[ham_idx[p * n_h:(p + 1) * n_h]].copy(); h["period"] = p; h["cluster"] = -1; parts.append(h)
        df = pd.concat(parts).reset_index(drop=True)
    else:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(X)
        comp = TruncatedSVD(1, random_state=seed).fit_transform(X)[:, 0]
        order = np.argsort([comp[km.labels_ == c].mean() for c in range(n_clusters)])
        rank = {int(c): r for r, c in enumerate(order)}
        spam["cluster"] = [rank[int(c)] for c in km.labels_]
        rows, remaining = [], spam.copy()
        for p in range(n_periods):
            mu = p * (n_clusters - 1) / max(n_periods - 1, 1)
            w = np.exp(-((remaining.cluster.to_numpy() - mu) ** 2) / (2 * sigma ** 2)) + 1e-9
            pick = rng.choice(remaining.index.to_numpy(), n_sp, replace=False, p=w / w.sum())
            sp_ = remaining.loc[pick].copy(); sp_["period"] = p
            h = ham.loc[ham_idx[p * n_h:(p + 1) * n_h]].copy(); h["period"] = p; h["cluster"] = -1
            rows += [sp_, h]; remaining = remaining.drop(pick)
        df = pd.concat(rows).reset_index(drop=True)
    df = _split_period0(df, seed)
    return _finalize(df, max_eval, seed)


def build_real_stream(csv_path, period_months=3, ads="exclude", min_per_class=25, seed=0, max_eval=None):
    """Real timestamped stream from the Kaggle 'PH Spam + Marketing SMS (w/ timestamps)' CSV.
    columns expected: date-received, category (spam|ads|gov|notifs|OTP), text."""
    raw = pd.read_csv(csv_path)
    low = {c.lower().replace("_", "-").strip(): c for c in raw.columns}
    dcol = low.get("date-received") or next(c for c in raw.columns if "date" in c.lower())
    ccol = low.get("category") or next(c for c in raw.columns if "categ" in c.lower() or "label" in c.lower())
    tcol = low.get("text") or next(c for c in raw.columns if "text" in c.lower() or "message" in c.lower())
    df = pd.DataFrame({"date": pd.to_datetime(raw[dcol], errors="coerce", utc=True),
                       "cat": raw[ccol].astype(str).str.strip().str.lower(), "text": raw[tcol]})
    n0 = len(df)
    df = df.dropna(subset=["date", "text"])
    df["text"] = df.text.astype(str).str.strip()
    df = df[(df.text.str.len() >= 5) & ~df.text.str.contains("redact", case=False)]
    print(f"[real] rows total={n0}, usable text+date={len(df)}")
    print(df.cat.value_counts().to_string())
    amap = {"exclude": None, "spam": 1, "ham": 0}[ads]
    lab = {"spam": 1, "gov": 0, "notifs": 0, "otp": 0, "ads": amap}
    df["label"] = df.cat.map(lab)
    df = df.dropna(subset=["label"]).copy(); df["label"] = df.label.astype(int)
    df = df.drop_duplicates(subset="text").sort_values("date")
    months = (df.date.dt.year - df.date.dt.year.min()) * 12 + df.date.dt.month
    df["pidx"] = ((months - months.min()) // period_months).astype(int)
    cnt = df.groupby(["pidx", "label"]).size().unstack(fill_value=0)
    print("per-period class counts:\n", cnt.to_string())
    good = cnt[(cnt.get(0, 0) >= min_per_class) & (cnt.get(1, 0) >= min_per_class)].index
    df = df[df.pidx.isin(good)].copy()
    if len(good) < 3:
        raise ValueError("Real dataset has fewer than 3 usable periods (ham text may be redacted or too few messages). "
                         "Use --mode simulated or lower --min_per_class / change --period_months.")
    df["period"] = df.pidx.map({p: i for i, p in enumerate(sorted(good))})
    df["id"] = np.arange(len(df)); df = df.reset_index(drop=True)
    df = _split_period0(df, seed)
    return _finalize(df[["id", "text", "label", "period", "split"]], max_eval, seed)
