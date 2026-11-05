"""Load Hetionet v1.0 and build sparse relation matrices.

Hetionet files (https://github.com/hetio/hetionet, folder hetnet/tsv):
  hetionet-v1.0-nodes.tsv     columns: id, name, kind
  hetionet-v1.0-edges.sif.gz  columns: source, metaedge, target   (stored with Git LFS)
Node ids look like "Compound::DB00014" / "Disease::DOID:1234".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Undirected relations that Hetionet stores only once
SYMMETRIC = {"CrC", "DrD", "GiG", "Gcr"}


def kind_of(node_id: str) -> str:
    return node_id.split("::")[0]


def load_hetionet(data_dir: str | Path):
    data_dir = Path(data_dir)
    nodes = pd.read_csv(data_dir / "hetionet-v1.0-nodes.tsv", sep="\t")
    edges = pd.read_csv(data_dir / "hetionet-v1.0-edges.sif.gz", sep="\t")
    return nodes, edges


@dataclass
class Relation:
    src_kind: str
    dst_kind: str
    src: np.ndarray  # integer index within src_kind
    dst: np.ndarray  # integer index within dst_kind


class HetGraph:
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        nodes = nodes.iloc[:, :3].copy()
        nodes.columns = ["id", "name", "kind"]
        nodes["kind"] = nodes["id"].map(kind_of)
        edges = edges.iloc[:, :3].copy()
        edges.columns = ["source", "metaedge", "target"]

        self.names = dict(zip(nodes["id"], nodes["name"]))
        self.ids = {k: g["id"].tolist() for k, g in nodes.groupby("kind", sort=True)}
        self.index = {nid: i for ids in self.ids.values() for i, nid in enumerate(ids)}
        self.rel: dict[str, Relation] = {}
        for me, g in edges.groupby("metaedge", sort=True):
            self.rel[me] = Relation(
                kind_of(g["source"].iloc[0]), kind_of(g["target"].iloc[0]),
                g["source"].map(self.index).to_numpy(), g["target"].map(self.index).to_numpy())

    def n(self, kind: str) -> int:
        return len(self.ids[kind])

    def name(self, kind: str, i: int) -> str:
        return self.names[self.ids[kind][i]]

    def matrix(self, me: str, keep: np.ndarray | None = None) -> sp.csr_matrix:
        """Binary src x dst adjacency. `keep` is an optional boolean mask over this relation's edges."""
        r = self.rel[me]
        s, d = (r.src, r.dst) if keep is None else (r.src[keep], r.dst[keep])
        A = sp.csr_matrix((np.ones(len(s)), (s, d)), shape=(self.n(r.src_kind), self.n(r.dst_kind)))
        A.data[:] = 1.0
        if me in SYMMETRIC:
            A = A.maximum(A.T)
        return A.tocsr()
