"""Filtered ranking + classification metrics for Compound x Disease score matrices."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(S: np.ndarray, pairs, known: np.ndarray, ks=(10, 50, 100)) -> dict:
    """S: (nCompound, nDisease) scores. pairs: (c_idx, d_idx) positives to evaluate.
    known: boolean matrix of ALL known positives (train+val+test+palliates); other known positives
    are removed from the candidate list ("filtered" setting) so they are not counted as errors."""
    c_idx, d_idx = pairs
    ranks = []
    for c, d in zip(c_idx, d_idx):
        mask = ~known[:, d]
        mask[c] = True
        cand, s = S[mask, d], S[c, d]
        ranks.append(1 + (cand > s).sum() + 0.5 * ((cand == s).sum() - 1))
    ranks = np.asarray(ranks, dtype=float)
    out = {"MRR": float(np.mean(1.0 / ranks)), "median_rank": float(np.median(ranks))}
    for k in ks:
        out[f"Hits@{k}"] = float(np.mean(ranks <= k))
    y = np.zeros(S.shape, dtype=bool)
    y[c_idx, d_idx] = True
    cand = (~known) | y
    out["AUROC"] = float(roc_auc_score(y[cand], S[cand]))
    out["AUPRC"] = float(average_precision_score(y[cand], S[cand]))
    return out
