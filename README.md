# Consensus AMRs as Fréchet Means in a Permutation-Invariant Graph Space

Code and data to reproduce the experiments of the paper. Given the AMRs of a set
of related sentences, the pipeline estimates their Fréchet mean in graph space,
projects it back to a well-formed AMR, and compares it with the selection
baselines on the 27 chapters of *The Little Prince*.

## Install

```bash
pip install -r requirements.txt
```

S2MATCH downloads GloVe vectors (about 130 MB) and the text-medoid baseline a
SentenceBERT encoder (about 90 MB) on first use.

## Run

```bash
python main.py search --jobs 8   # a Fréchet mean per chapter, then tune (s, tau)
python main.py aggregate         # turn the search into the reported tables
```

`search` takes about ninety minutes on 9 cores, nearly all of it in the means.
They are cached under `results/raw/means/`, so a second search over other
thresholds takes minutes. `aggregate` runs in seconds and loads no model.

`--s-range`, `--tau-range`, `--trials` and `--restarts` change the search; every
setting used is recorded in `results/raw/config.json`.

## Outputs

| File | Content |
|---|---|
| `results/experiment_scores.csv` | one row per experiment × metric × baseline: the two scores, the gain, its p-values |
| `results/selected_params.csv` | the (s, tau) chosen for each chapter |
| `results/consensus_graph_smatch.csv` | the 27 consensus AMRs, SMATCH-selected |
| `results/consensus_graph_s2match.csv` | the same, S2MATCH-selected |

`results/raw/` keeps every score and every graph of the sweep.

The three experiments differ only in how the two thresholds are chosen:

- **main** — tuned on the evaluated chapter (Table 1);
- **global_in_sample** — one configuration for the whole corpus (Table 2);
- **loco** — leave-one-chapter-out: chosen on 26 chapters, read on the 27th.

## Code

| Module | Content |
|---|---|
| `consensus/corpus.py` | AMR files to tensor encodings, and back to Penman |
| `consensus/align.py` | Frank-Wolfe alignment, the quotient distance |
| `consensus/mean.py` | mean-AAC |
| `consensus/project.py` | the constrained decoder |
| `consensus/metrics.py` | SMATCH and S2MATCH, via smatchpp |
| `consensus/baselines.py` | the three selection baselines |
| `consensus/search.py` | the Optuna campaigns |
| `consensus/aggregate.py` | the three experiments |

## Data

`data/petit_prince/` holds the 27 chapters of the AMR-annotated *Little Prince*
(AMR release 3.0, <https://amr.isi.edu>), one file per chapter. Preprocessing
keeps instantiated concepts only and drops the graphs above 15 nodes, leaving
1,465 of them.
