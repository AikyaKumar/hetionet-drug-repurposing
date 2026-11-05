"""Degree-weighted metapath features for Compound -> Disease pairs (a simplified DWPC).

Each metapath is a list of (metaedge, reversed?) steps. Path weights are damped by node degree
(exponent w, 0.4 as in the Hetionet paper). This counts walks, not strictly simple paths: a
documented simplification.

LEAKAGE CONTROL: `keep_ctd` is a boolean mask over the Compound-treats-Disease edges that the
features are allowed to see. Held-out (validation/test) treats edges are masked out before ANY
path is computed, including paths that pass through CtD (CrCtD, CtDrD, CcSEcCtD).
The Compound-palliates-Disease (CpD) relation is never used as a feature.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp

from .graph import HetGraph

F, R = False, True
METAPATHS = {
    "CbGaD":      [("CbG", F), ("DaG", R)],
    "CuGdD":      [("CuG", F), ("DdG", R)],
    "CdGuD":      [("CdG", F), ("DuG", R)],
    "CuGuD":      [("CuG", F), ("DuG", R)],
    "CdGdD":      [("CdG", F), ("DdG", R)],
    "CbGiGaD":    [("CbG", F), ("GiG", F), ("DaG", R)],
    "CbGpPWpGaD": [("CbG", F), ("GpPW", F), ("GpPW", R), ("DaG", R)],
    "CrCtD":      [("CrC", F), ("CtD", F)],
    "CtDrD":      [("CtD", F), ("DrD", F)],
    "CcSEcCtD":   [("CcSE", F), ("CcSE", R), ("CtD", F)],
}


def _damp(A: sp.csr_matrix, w: float) -> sp.csr_matrix:
    r = np.asarray(A.sum(1)).ravel()
    c = np.asarray(A.sum(0)).ravel()
    rinv = np.where(r > 0, np.power(np.maximum(r, 1), -w), 0.0)
    cinv = np.where(c > 0, np.power(np.maximum(c, 1), -w), 0.0)
    return sp.diags(rinv) @ A @ sp.diags(cinv)


def metapath_features(g: HetGraph, keep_ctd: np.ndarray, w: float = 0.4):
    """Return (features[nCompound, nDisease, K], names)."""
    cache: dict[str, sp.csr_matrix] = {}
    mats, names = [], []
    for name, steps in METAPATHS.items():
        missing = [me for me, _ in steps if me not in g.rel]
        if missing:
            warnings.warn(f"metapath {name} skipped, missing relation(s): {missing}")
            continue
        M = None
        for me, rev in steps:
            if me not in cache:
                cache[me] = g.matrix(me, keep_ctd if me == "CtD" else None)
            A = cache[me].T.tocsr() if rev else cache[me]
            A = _damp(A, w)
            M = A if M is None else M @ A
        M = M.toarray()
        pos = M[M > 0]
        scale = pos.mean() if pos.size else 1.0
        mats.append(np.log1p(M / scale))
        names.append(name)
    return np.stack(mats, axis=-1), names
