"""Baselines and the main metapath model. All of them see only the TRAINING graph."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import evaluate


def sample_negatives(train_known: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Random unlabeled pairs. Only TRAINING labels are used to filter (test labels are never touched)."""
    rng = np.random.RandomState(seed)
    pool = np.flatnonzero(~train_known.ravel())
    return rng.choice(pool, size=n, replace=False)


def _fit_scores(make_clf, F, pos_flat, neg_flat):
    X = F.reshape(-1, F.shape[-1])
    idx = np.r_[pos_flat, neg_flat]
    y = np.r_[np.ones(len(pos_flat)), np.zeros(len(neg_flat))]
    clf = make_clf().fit(X[idx], y)
    return clf.predict_proba(X)[:, 1].reshape(F.shape[:2]), clf


def logreg(C):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def hgb(seed):
    return lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)


def run_all(F, names, train_pairs, val_pairs, test_pairs, known, train_known, seed, neg_ratio=10):
    """Return {method: test metrics}; hyper-parameters are chosen on the validation split only."""
    nC, nD = F.shape[:2]
    out = {}
    rng = np.random.RandomState(seed)

    # 1. random
    S = rng.rand(nC, nD)
    out["random"] = evaluate(S, test_pairs, known)

    # 2. popularity: number of training treats-edges of the compound
    deg = np.bincount(train_pairs[0], minlength=nC).astype(float)
    out["popularity (compound degree)"] = evaluate(np.repeat(deg[:, None], nD, 1), test_pairs, known)

    # 3. best single metapath, chosen on validation
    val_mrr = [evaluate(F[:, :, k], val_pairs, known)["MRR"] for k in range(F.shape[-1])]
    k = int(np.argmax(val_mrr))
    out[f"best single metapath ({names[k]})"] = evaluate(F[:, :, k], test_pairs, known)

    # 4/5. supervised models on all metapath features
    pos_flat = np.ravel_multi_index(train_pairs, (nC, nD))
    neg_flat = sample_negatives(train_known, neg_ratio * len(pos_flat), seed)
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        S, clf = _fit_scores(logreg(C), F, pos_flat, neg_flat)
        m = evaluate(S, val_pairs, known)["MRR"]
        if best is None or m > best[0]:
            best = (m, C, S, clf)
    out[f"logistic regression, all metapaths (C={best[1]})"] = evaluate(best[2], test_pairs, known)
    S, _ = _fit_scores(hgb(seed), F, pos_flat, neg_flat)
    out["gradient boosting, all metapaths"] = evaluate(S, test_pairs, known)
    return out
