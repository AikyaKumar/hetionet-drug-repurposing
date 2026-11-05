import numpy as np


def ctd_pairs(g):
    r = g.rel["CtD"]
    return r.src, r.dst


def split_edges(n_edges: int, seed: int, frac=(0.8, 0.1, 0.1)):
    """Random train/val/test split of the Compound-treats-Disease edges (by edge)."""
    perm = np.random.RandomState(seed).permutation(n_edges)
    n_tr, n_va = int(frac[0] * n_edges), int(frac[1] * n_edges)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def pair_matrix(shape, c, d):
    M = np.zeros(shape, dtype=bool)
    M[c, d] = True
    return M


def cpd_matrix(g, shape):
    """Palliates edges: not used as features; excluded from negatives and from ranking candidates."""
    if "CpD" not in g.rel:
        return np.zeros(shape, dtype=bool)
    r = g.rel["CpD"]
    return pair_matrix(shape, r.src, r.dst)
