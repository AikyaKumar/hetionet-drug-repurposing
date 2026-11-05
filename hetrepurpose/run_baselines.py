"""python -m hetrepurpose.run_baselines --data-dir data --seeds 5 --out results"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import run_all
from .graph import HetGraph, load_hetionet
from .metapaths import metapath_features
from .splits import cpd_matrix, ctd_pairs, pair_matrix, split_edges


def run(g: HetGraph, seeds: int, out_dir: Path):
    nC, nD = g.n("Compound"), g.n("Disease")
    c_all, d_all = ctd_pairs(g)
    cpd = cpd_matrix(g, (nC, nD))
    known = pair_matrix((nC, nD), c_all, d_all) | cpd
    print(f"{nC} compounds, {nD} diseases, {len(c_all)} treats edges")
    rows = []
    for seed in range(seeds):
        tr, va, te = split_edges(len(c_all), seed)
        keep = np.zeros(len(c_all), dtype=bool)
        keep[tr] = True                       # features may only see TRAINING treats edges
        F, names = metapath_features(g, keep)
        pairs = lambda idx: (c_all[idx], d_all[idx])
        train_known = pair_matrix((nC, nD), *pairs(tr)) | cpd
        res = run_all(F, names, pairs(tr), pairs(va), pairs(te), known, train_known, seed)
        for method, m in res.items():
            rows.append({"seed": seed, "method": method, **m})
        print(f"seed {seed} done")
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "baselines_per_seed.csv", index=False)
    metrics = ["AUROC", "AUPRC", "MRR", "Hits@10", "Hits@50", "Hits@100"]
    agg = df.assign(method=df.method.str.replace(r" \(C=.*\)", "", regex=True)
                    .str.replace(r"\(.*\)", "", regex=True).str.strip()).groupby("method")[metrics].agg(["mean", "std"])
    table = pd.DataFrame({m: agg[m]["mean"].map("{:.3f}".format) + " ± " + agg[m]["std"].map("{:.3f}".format) for m in metrics})
    table.to_csv(out_dir / "baselines_summary.csv")
    (out_dir / "baselines_summary.md").write_text(table.to_markdown())
    print(table.to_string())
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    nodes, edges = load_hetionet(a.data_dir)
    run(HetGraph(nodes, edges), a.seeds, Path(a.out))
