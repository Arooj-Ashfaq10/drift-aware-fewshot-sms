import numpy as np
from scipy.stats import binomtest


def confusion(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(int)
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    return tp, fp, tn, fn


def metrics(y, p):
    tp, fp, tn, fn = confusion(y, p)
    d = lambda a, b: a / b if b > 0 else 0.0
    prec = d(tp, tp + fp); rec = d(tp, tp + fn)
    return dict(
        acc=d(tp + tn, tp + fp + tn + fn), precision=prec, recall=rec,
        f1=d(2 * prec * rec, prec + rec), tnr=d(tn, tn + fp),
        fpr=d(fp, fp + tn), fnr=d(fn, fn + tp), n=tp + fp + tn + fn,
    )


def bootstrap_ci(y, p, key="f1", n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metrics(y[idx], p[idx])[key])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y, p1, p2):
    """Exact McNemar test on paired predictions. Returns (b, c, p_value);
    b = only system 1 correct, c = only system 2 correct."""
    y = np.asarray(y); c1 = (np.asarray(p1) == y); c2 = (np.asarray(p2) == y)
    b = int((c1 & ~c2).sum()); c = int((~c1 & c2).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5).pvalue)
