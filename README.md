# Consensus AMR: Estimating Central Semantic Graphs via Fréchet Means in Graph Space

Code and data to reproduce the experiments of the paper. Given the AMRs of a set of related sentences, the pipeline estimates their Fréchet mean in graph space which are called semantic means, projects it back to a well-formed AMR, and compares it with the selection baselines on the 27 chapters of *The Little Prince*.

## Install

```bash
pip install -r requirements.txt
```

S2MATCH downloads GloVe vectors (about 130 MB) and the text-medoid baseline a
SentenceBERT encoder (about 90 MB) on first use.

## Run

```bash
python main.py search --jobs 8   # a Fréchet mean per chapter, then sweep τ
python main.py aggregate         # turn the search into the reported tables
python main.py all --jobs 8      # both
```

`search` takes about ninety minutes with `--jobs 8`. Cached semantic mean under `results/means/`, so a second search over other values of τ takes about ten minutes. `aggregate` runs in seconds. 

`--threshold-range` and `--threshold-step` move the grid of τ, `--restarts` the number of mean-AAC runs per chapter; every setting used is recorded in `results/config.json`.

## Outputs


| File                                  | Content                                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| `results/experiment_scores.csv`       | one row per experiment × metric × baseline: the two scores, the gain, its p-values |
| `results/selected_params.csv`         | the τ each experiment and metric applies to each chapter                           |
| `results/consensus_graph_smatch.csv`  | the 27 consensus AMRs of *main*, SMATCH-selected                                   |
| `results/consensus_graph_s2match.csv` | the same, S2MATCH-selected                                                         |


`search` generates `results/scores.csv.gz` and `results/graphs.csv.gz`, which store every score and every graph produced by the sweep, respectively. It also caches the semantic means in `results/means/`.

The three experiments differ only in how τ is chosen:

- **main** — tuned on the evaluated chapter (Table 1);
- **global_in_sample** — a single τ for the whole corpus (Table 2); (not reported in the paper)
- **loco** — leave-one-chapter-out: chosen on 26 chapters, read on the 27th.



## Code


| Module                   | Content                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| `consensus/corpus.py`    | AMR files to tensor encodings, and back to Penman                            |
| `consensus/align.py`     | Frank-Wolfe alignment, the quotient distance                                 |
| `consensus/mean.py`      | mean-AAC                                                                     |
| `consensus/project.py`   | the constrained decoder: connectivity, role determinism, core-role licensing |
| `consensus/metrics.py`   | SMATCH and S2MATCH, via smatchpp                                             |
| `consensus/baselines.py` | the three selection baselines                                                |
| `consensus/search.py`    | the sweep over τ: one study per chapter, one over the corpus                 |
| `consensus/aggregate.py` | experimental results                                                         |




## Data

`data/petit_prince/` holds the 27 chapters of the AMR-annotated *Little Prince*
(AMR release 3.0, [https://amr.isi.edu](https://amr.isi.edu)), one file per chapter. Preprocessing
keeps instantiated concepts only, and drops each chapter's heading graph and the
graphs above 15 nodes, leaving 1,465 of them.