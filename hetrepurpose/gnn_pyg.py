"""STAGE 2 (optional): heterogeneous GraphSAGE link predictor with PyTorch Geometric.

!! Not executed in the environment where this repo was written (torch / PyG were not available there).
!! Run it first with --epochs 5 --hidden 16 to check shapes, then scale up. Everything it shares with
!! stage 1 (graph loading, split, leakage masking, evaluation) IS covered by the tests.

python -m hetrepurpose.gnn_pyg --data-dir data --seeds 3 --epochs 200 --hidden 64 --out results

Leakage control: the message-passing graph is built WITHOUT the validation/test Compound-treats-Disease
edges (and their reverse edges), exactly like the metapath features in stage 1.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

from .evaluate import evaluate
from .graph import HetGraph, load_hetionet
from .splits import cpd_matrix, ctd_pairs, pair_matrix, split_edges


def _t(kind: str) -> str:
    return re.sub(r"\W", "_", kind)


def build_data(g: HetGraph, keep_ctd: np.ndarray) -> HeteroData:
    data = HeteroData()
    for kind, ids in g.ids.items():
        data[_t(kind)].num_nodes = len(ids)
    for me, r in g.rel.items():
        s, d = (r.src[keep_ctd], r.dst[keep_ctd]) if me == "CtD" else (r.src, r.dst)
        data[_t(r.src_kind), re.sub(r"\W", "_", me), _t(r.dst_kind)].edge_index = torch.tensor(np.vstack([s, d]), dtype=torch.long)
    return T.ToUndirected()(data)


class Encoder(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.c1, self.c2 = SAGEConv((-1, -1), hidden), SAGEConv((-1, -1), hidden)

    def forward(self, x, edge_index):
        return self.c2(self.c1(x, edge_index).relu(), edge_index)


class Net(torch.nn.Module):
    def __init__(self, data, hidden):
        super().__init__()
        self.emb = torch.nn.ModuleDict({t: torch.nn.Embedding(data[t].num_nodes, hidden) for t in data.node_types})
        self.enc = to_hetero(Encoder(hidden), data.metadata(), aggr="sum")

    def forward(self, data):
        x = {t: self.emb[t].weight for t in data.node_types}
        return self.enc(x, data.edge_index_dict)


def run_seed(g, seed, epochs, hidden, lr, device):
    torch.manual_seed(seed); np.random.seed(seed)
    nC, nD = g.n("Compound"), g.n("Disease")
    c_all, d_all = ctd_pairs(g)
    known = pair_matrix((nC, nD), c_all, d_all) | cpd_matrix(g, (nC, nD))
    tr, va, te = split_edges(len(c_all), seed)
    keep = np.zeros(len(c_all), dtype=bool); keep[tr] = True
    data = build_data(g, keep).to(device)
    model = Net(data, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pc, pd_ = torch.tensor(c_all[tr], device=device), torch.tensor(d_all[tr], device=device)
    pairs = lambda idx: (c_all[idx], d_all[idx])
    best = (-1.0, None)
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        z = model(data)
        zc, zd = z["Compound"], z["Disease"]
        pos = (zc[pc] * zd[pd_]).sum(-1)
        neg = (zc[pc] * zd[torch.randint(0, nD, (len(pc),), device=device)]).sum(-1)   # corrupt the disease
        loss = Fn.binary_cross_entropy_with_logits(torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
        loss.backward(); opt.step()
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                z = model(data)
                S = (z["Compound"] @ z["Disease"].T).cpu().numpy()
            val = evaluate(S, pairs(va), known)["MRR"]          # model selection on validation only
            if val > best[0]:
                best = (val, S)
            print(f"seed {seed} epoch {epoch:4d} loss {loss.item():.4f} val MRR {val:.3f}")
    return evaluate(best[1], pairs(te), known)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = HetGraph(*load_hetionet(a.data_dir))
    rows = [{"seed": s, "method": "heterogeneous GraphSAGE", **run_seed(g, s, a.epochs, a.hidden, a.lr, device)} for s in range(a.seeds)]
    df = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "gnn_per_seed.csv", index=False)
    print(df.drop(columns="method").describe().loc[["mean", "std"]].round(3).to_string())
