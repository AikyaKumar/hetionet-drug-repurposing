# hetionet-drug-repurposing
Drug repurposing using biomedical knowledge graphs and graph neural networks.
# Drug Repurposing on Hetionet: leakage-safe graph ML

> **Status: in progress** (started 10/2026). Pipeline and tests are in place.

**In one paragraph:** I'm predicting which drugs may treat which diseases (*Compound–treats–Disease* links) in [Hetionet](https://github.com/hetio/hetionet), a public biomedical knowledge graph, and comparing simple baselines, metapath-feature models and graph neural networks fairly. The focus is on getting the evaluation right: no leakage, several seeds, honest metrics, and explanations a pharmacologist can read.

## Why I'm building this
My Master's thesis applied GNNs to airline networks. I'm applying the same skills (graph ML, rigorous held-out evaluation) to biomedical networks, in the drug repurposing setting.

## Status
- [ ] Data loading, leakage-safe splits and metapath features
- [ ] Leakage and metric tests passing (`pytest tests -q`)
- [ ] Baselines run on real Hetionet data (5 seeds)
- [ ] CNS case study with path-based explanations
- [ ] Heterogeneous GNN (PyTorch Geometric) added to the comparison
- [ ] Results table and discussion below

## Approach
| Method | Uses |
|---|---|
| popularity | number of known treatments of a compound |
| best single metapath | one degree-weighted path type, chosen on validation |
| logistic regression | all metapath features (interpretable) |
| gradient boosting | all metapath features |
| heterogeneous GraphSAGE *(planned)* | the whole graph, learned embeddings |

Metapath features are a simplified degree-weighted path count, e.g. `CbGaD` = compound–binds–gene–associates–disease (counts walks, not strictly simple paths).

## Evaluation protocol
- 10% validation / 10% test of the treatment edges; **5 random splits (seeds)**, reporting mean ± std.
- **Filtered ranking** of the true compound per held-out pair (MRR, Hits@10/50/100) plus AUROC and AUPRC (AUPRC matters most: positives are rare).
- Hyper-parameters chosen on validation only; the test split is used once.

## Leakage controls (tested in `tests/`)
1. Held-out treatment edges are removed **before** any feature is computed, including paths that pass through treatment edges. A test checks that masking them equals deleting them from the graph, and that the check can fail.
2. Palliative edges are never used as features, and are excluded from negatives and candidates.
3. Negatives are sampled using training labels only.
4. Splits are disjoint and complete; metrics are checked on perfect and random scores.

## Results
*Coming soon...*

## Case study (planned)
Rank not-yet-known compounds for CNS diseases (e.g. Parkinson's, schizophrenia, epilepsy) and show the paths and genes behind each score. These are model hypotheses to check against the literature, **not clinical claims**.

## Run it
```bash
pip install -r requirements.txt
# Download from github.com/hetio/hetionet (folder hetnet/tsv) into ./data/ :
#   hetionet-v1.0-nodes.tsv and hetionet-v1.0-edges.sif.gz (edge file uses Git LFS)
pytest tests -q
python -m hetrepurpose.run_baselines --data-dir data --seeds 5 --out results
python -m hetrepurpose.case_study --data-dir data --out results/cns_case_study.md
```

## Limitations
- About 750 known treatment edges: results will vary by split, hence several seeds.
- Unlabeled pairs are not true negatives; the graph is biased toward well-studied drugs and diseases.
- Random edge split; hiding whole diseases (cold start) would be harder and more realistic.

## Data and reference
Hetionet v1.0 (CC0): Himmelstein et al., *Systematic integration of biomedical knowledge prioritizes drugs for repurposing*, eLife 2017.
