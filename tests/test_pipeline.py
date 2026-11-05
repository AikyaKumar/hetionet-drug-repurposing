import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import make_synthetic  # noqa: E402
from hetrepurpose.evaluate import evaluate  # noqa: E402
from hetrepurpose.graph import HetGraph  # noqa: E402
from hetrepurpose.metapaths import metapath_features  # noqa: E402
from hetrepurpose.splits import cpd_matrix, ctd_pairs, pair_matrix, split_edges  # noqa: E402


def _graph():
    return make_synthetic(seed=1)


def test_splits_are_disjoint_and_complete():
    tr, va, te = split_edges(755, seed=3)
    assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == 755


def test_heldout_edges_do_not_influence_features():
    """Masking held-out treats edges must equal physically deleting them from the graph."""
    nodes, edges = _graph()
    g = HetGraph(nodes, edges)
    c, d = ctd_pairs(g)
    tr, va, te = split_edges(len(c), seed=0)
    keep = np.zeros(len(c), dtype=bool)
    keep[tr] = True
    F_masked, names = metapath_features(g, keep)

    ctd_rows = np.flatnonzero(edges.metaedge.values == "CtD")
    dropped = edges.drop(index=edges.index[ctd_rows[np.concatenate([va, te])]])
    g2 = HetGraph(nodes, dropped)
    F_deleted, names2 = metapath_features(g2, np.ones(len(g2.rel["CtD"].src), dtype=bool))
    assert names == names2
    assert np.allclose(F_masked, F_deleted)


def test_features_change_if_heldout_edges_are_NOT_masked():
    """Sanity check that the leakage test above is able to fail."""
    nodes, edges = _graph()
    g = HetGraph(nodes, edges)
    c, d = ctd_pairs(g)
    tr, va, te = split_edges(len(c), seed=0)
    keep = np.zeros(len(c), dtype=bool)
    keep[tr] = True
    F_masked, _ = metapath_features(g, keep)
    F_full, _ = metapath_features(g, np.ones(len(c), dtype=bool))
    assert not np.allclose(F_masked, F_full)


def test_perfect_scores_give_perfect_metrics():
    nodes, edges = _graph()
    g = HetGraph(nodes, edges)
    c, d = ctd_pairs(g)
    shape = (g.n("Compound"), g.n("Disease"))
    known = pair_matrix(shape, c, d) | cpd_matrix(g, shape)
    S = known.astype(float)
    m = evaluate(S, (c, d), known)
    assert m["MRR"] == 1.0 and m["AUROC"] == 1.0


def test_random_scores_are_near_chance():
    nodes, edges = _graph()
    g = HetGraph(nodes, edges)
    c, d = ctd_pairs(g)
    shape = (g.n("Compound"), g.n("Disease"))
    known = pair_matrix(shape, c, d) | cpd_matrix(g, shape)
    m = evaluate(np.random.RandomState(0).rand(*shape), (c, d), known)
    assert 0.45 < m["AUROC"] < 0.55


def test_smoke_supervised_beats_random_on_planted_structure():
    from hetrepurpose.run_baselines import run
    nodes, edges = _graph()
    table = run(HetGraph(nodes, edges), seeds=2, out_dir=Path("/tmp/hetio_smoke"))
    mrr = table["MRR"].map(lambda s: float(s.split(" ")[0]))
    assert mrr["logistic regression, all metapaths"] > 3 * mrr["random"]
