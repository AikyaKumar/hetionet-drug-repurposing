"""Tiny synthetic graph in Hetionet format (planted topic structure). For tests/smoke runs only."""
import numpy as np
import pandas as pd


def make_synthetic(seed=0, nC=300, nD=40, nG=600, nSE=60, nPW=30, topics=8):
    rng = np.random.RandomState(seed)
    ct, dt, gt = rng.randint(0, topics, nC), rng.randint(0, topics, nD), rng.randint(0, topics, nG)
    cid = [f"Compound::DB{i:05d}" for i in range(nC)]
    did = [f"Disease::DOID:{i}" for i in range(nD)]
    gid = [f"Gene::{i}" for i in range(nG)]
    sid = [f"Side Effect::C{i:07d}" for i in range(nSE)]
    pid = [f"Pathway::PC7_{i}" for i in range(nPW)]
    nodes = pd.DataFrame({"id": cid + did + gid + sid + pid,
                          "name": [f"compound{i}" for i in range(nC)] + [f"disease{i}" for i in range(nD)] +
                                  [f"gene{i}" for i in range(nG)] + [f"se{i}" for i in range(nSE)] + [f"pw{i}" for i in range(nPW)],
                          "kind": ["Compound"] * nC + ["Disease"] * nD + ["Gene"] * nG + ["Side Effect"] * nSE + ["Pathway"] * nPW})
    E = []

    def add(me, src_ids, dst_ids, p_same, p_diff, st, dtp):
        for i, s in enumerate(src_ids):
            for j, d in enumerate(dst_ids):
                if me in ("CrC", "DrD", "GiG") and j <= i and src_ids is dst_ids:
                    continue
                p = p_same if st[i] == dtp[j] else p_diff
                if rng.rand() < p:
                    E.append((s, me, d))

    add("CbG", cid, gid, 0.05, 0.002, ct, gt)
    add("DaG", did, gid, 0.10, 0.003, dt, gt)
    add("CuG", cid, gid, 0.03, 0.002, ct, gt)
    add("CdG", cid, gid, 0.03, 0.002, ct, gt)
    add("DuG", did, gid, 0.05, 0.003, dt, gt)
    add("DdG", did, gid, 0.05, 0.003, dt, gt)
    add("CtD", cid, did, 0.20, 0.004, ct, dt)
    add("CpD", cid, did, 0.02, 0.002, ct, dt)
    add("CrC", cid, cid, 0.04, 0.002, ct, ct)
    add("DrD", did, did, 0.15, 0.01, dt, dt)
    add("GiG", gid, gid, 0.02, 0.001, gt, gt)
    st_se, st_pw = rng.randint(0, topics, nSE), rng.randint(0, topics, nPW)
    add("CcSE", cid, sid, 0.15, 0.03, ct, st_se)
    add("GpPW", gid, pid, 0.10, 0.01, gt, st_pw)
    edges = pd.DataFrame(E, columns=["source", "metaedge", "target"])
    return nodes, edges
