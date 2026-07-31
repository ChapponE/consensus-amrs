# Consensus AMRs as Fréchet Means in a Permutation-Invariant Graph Space

Code and data to reproduce the experiments of the paper. Given the AMRs of a set
of related sentences, the pipeline estimates their Fréchet mean in graph space
and projects it back to a well-formed AMR, then compares it with the selection
baselines on the 27 chapters of *The Little Prince*.

## Install

```bash
pip install -r requirements.txt
```

S2MATCH downloads GloVe vectors (~130 MB) and the text-medoid baseline a
SentenceBERT encoder (~90 MB) on first use.

## Run

```bash
python main.py search --jobs 8   # search the 27 chapters over (s, tau)
python main.py aggregate         # turn the search into the reported tables
```

`search` computes one Fréchet mean per chapter, then runs two Optuna campaigns
over the node-activation threshold `s` and the edge threshold `tau`: one study
per chapter, and one study over the whole corpus. Both maximise SMATCH and
S2MATCH jointly, 40 seeded trials each, `s` in [0.05, 0.20] and `tau` in
[0.15, 0.25]. Every consensus graph is scored against every input of its
chapter, as are the three selection baselines. The per-input scores and the
consensus graphs go to `results/raw/`. Most of the runtime is the means, about
ninety minutes for the corpus on 9 cores; they are cached under
`results/raw/means/`, so a second search over other thresholds takes minutes.
`aggregate` reads the result back in a few seconds and loads no model.

`--s-range`, `--tau-range` and `--trials` move the searched window and its
budget, and `--restarts` the number of mean-AAC runs per chapter. All of them
are recorded in `results/raw/config.json`.

## Outputs

| File | Content |
|---|---|
| `results/experiment_scores.csv` | one row per experiment × metric × baseline: both scores, the gain, its p-values, and how many chapters improve |
| `results/selected_params.csv` | the (s, tau) selected by each experiment, per chapter |
| `results/consensus_graph_smatch.csv` | the 27 consensus AMRs of the main experiment, SMATCH-selected |
| `results/consensus_graph_s2match.csv` | the same, S2MATCH-selected |
| `results/raw/scores.csv.gz` | every consensus × input score of the sweep |
| `results/raw/graphs.csv.gz` | every consensus graph of the sweep |

Three experiments differ only in how the two projector thresholds are chosen,
per metric, ties going to the smallest `s` then the smallest `tau`:

- **main** — tuned on the evaluated chapter, from its own study (in-sample, Table 1);
- **global_in_sample** — the corpus study's best configuration (Table 2);
- **loco** — leave-one-chapter-out: the corpus study's best configuration over 26
  chapters, read on the 27th.

The consensus is compared with the medoid under SMATCH, the medoid under
S2MATCH, and the text medoid. `gain` is the mean over chapters of the
difference, `p_ttest` a one-sided paired t-test over the 27 chapters, `p_sign`
the matching sign test, and `n_sig_better` / `n_sig_worse` count the chapters
where the same test over the chapter's own sentences is significant.

## Code

| Module | Content |
|---|---|
| `consensus/corpus.py` | AMR files to per-chapter tensor encodings, and the single tensor → Penman decoder |
| `consensus/align.py` | Frank-Wolfe alignment, the quotient distance of Section 4 |
| `consensus/mean.py` | mean-AAC, Algorithm 1 |
| `consensus/project.py` | node activation, edge scores, MSCG decoding, role determinism, concept labelling |
| `consensus/metrics.py` | SMATCH and S2MATCH, via smatchpp |
| `consensus/baselines.py` | the three selection baselines |
| `consensus/search.py` | the two Optuna campaigns |
| `consensus/aggregate.py` | the three experiments |

Everything upstream of the two thresholds is fixed: 30 mean-AAC iterations at
tolerance 1e-4, each alignment solved by Frank-Wolfe with 30 iterations from two
initialisations, and 50 subgradient iterations for the determinism dual. The
alternation only converges locally, so it is restarted from 20 points spread
over the population and the run of lowest Fréchet energy is kept.

## Data

`data/petit_prince/` holds the 27 chapters of the AMR-annotated *Little Prince*
(AMR release 3.0, <https://amr.isi.edu>), one file per chapter. Preprocessing
keeps only instantiated concepts, drops the chapter heading of each file and the
graphs above 15 nodes, and leaves 1,465 graphs.
