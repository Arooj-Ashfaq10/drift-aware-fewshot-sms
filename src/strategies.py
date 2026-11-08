"""Example selection (recent vs random), refresh windows and the label-free drift monitor."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

_HV = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 16, alternate_sign=False, norm="l2")


def pick_examples(pool, k=None, seed=0, n_spam=None, n_ham=None):
    """Class-balanced random draw from `pool` (DataFrame with text,label). Default: k/2 spam + k/2 ham."""
    if n_spam is None:
        n_spam, n_ham = k // 2, k - k // 2
    rng = np.random.default_rng(seed)
    sp = pool[pool.label == 1]; hm = pool[pool.label == 0]
    sp = sp.iloc[rng.permutation(len(sp))[:n_spam]]; hm = hm.iloc[rng.permutation(len(hm))[:n_ham]]
    ex = pd.concat([sp, hm]); ex = ex.iloc[rng.permutation(len(ex))]
    return list(zip(ex.text.tolist(), ex.label.tolist()))


def window(stream, p):
    """Labelled messages of period p (for period 0 only the 'pool' part is labelled/available)."""
    w = stream[stream.period == p]
    return w[w.split == "pool"] if p == 0 else w


def history(stream, t):
    """All labelled messages from periods < t."""
    h = stream[stream.period < t]
    return h[~((h.period == 0) & (h.split == "eval"))]


def centroid(texts):
    c = np.asarray(_HV.transform(texts).mean(axis=0)).ravel()
    return c / (np.linalg.norm(c) + 1e-12)


def drift_distance(ref_texts, cur_texts):
    """1 - cosine similarity between character n-gram centroids of two sets of (predicted-spam) messages."""
    return 1.0 - float(centroid(ref_texts) @ centroid(cur_texts))


def calibrate_tau(texts, z=3.0, n=100, seed=0):
    """Threshold = mean + z*std of the distance between random halves of one period (pure sampling noise)."""
    rng = np.random.default_rng(seed); texts = list(texts); d = []
    for _ in range(n):
        perm = rng.permutation(len(texts)); h = len(texts) // 2
        d.append(drift_distance([texts[i] for i in perm[:h]], [texts[i] for i in perm[h:]]))
    return float(np.mean(d) + z * np.std(d))
