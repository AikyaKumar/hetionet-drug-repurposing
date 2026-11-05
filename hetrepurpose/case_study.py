"""Rank unseen compounds for CNS diseases and explain each hit.
python -m hetrepurpose.case_study --data-dir data --out results/cns_case_study.md

These are model-generated HYPOTHESES for sanity-checking against the literature, not clinical claims.
"""
import argparse
from pathlib import Path

import numpy as np

from .baselines import _fit_scores, logreg, sample_negatives
from .graph import HetGraph, load_hetionet
from .metapaths import metapath_features
from .splits import cpd_matrix, ctd_pairs, pair_matrix

CNS_KEYWORDS = ["parkinson", "schizophrenia", "epilep", "alzheimer", "bipolar", "major depress",
                "multiple sclerosis", "migraine", "huntington", "autis"]


def match_diseases(g, keywords=CNS_KEYWORDS):
    hits = {}
    for i, nid in enumerate(g.ids["Disease"]):
        nm = g.names[nid].lower()
        if any(k in nm for k in keywords):
            hits[i] = g.names[nid]
    return hits


def shared_genes(g, c, d, top=5):
    cb = g.matrix("CbG")[c].toarray().ravel() > 0
    da = g.matrix("DaG")[d].toarray().ravel() > 0
    idx = np.flatnonzero(cb & da)[:top]
    return [g.name("Gene", i) for i in idx]


def main(data_dir, out, topk=10, seed=0, C=1.0):
    g = HetGraph(*load_hetionet(data_dir))
    nC, nD = g.n("Compound"), g.n("Disease")
    c_all, d_all = ctd_pairs(g)
    cpd = cpd_matrix(g, (nC, nD))
    known = pair_matrix((nC, nD), c_all, d_all) | cpd
    F, names = metapath_features(g, np.ones(len(c_all), dtype=bool))   # all known treats edges
    pos = np.ravel_multi_index((c_all, d_all), (nC, nD))
    neg = sample_negatives(known, 10 * len(pos), seed)
    S, clf = _fit_scores(logreg(C), F, pos, neg)
    scaler, lr = clf.steps[0][1], clf.steps[1][1]
    lines = ["# CNS case study (model hypotheses, not clinical claims)\n"]
    for d, dname in match_diseases(g).items():
        order = [c for c in np.argsort(-S[:, d]) if not known[c, d]][:topk]
        lines.append(f"\n## {dname}\n")
        for rank, c in enumerate(order, 1):
            contrib = lr.coef_[0] * scaler.transform(F[c, d][None])[0]
            top = [names[i] for i in np.argsort(-contrib)[:3]]
            genes = ", ".join(shared_genes(g, c, d)) or "none"
            lines.append(f"{rank}. **{g.name('Compound', c)}** (score {S[c, d]:.3f}); top metapaths: {', '.join(top)}; shared genes (CbG-DaG): {genes}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/cns_case_study.md")
    a = ap.parse_args()
    main(a.data_dir, a.out)
