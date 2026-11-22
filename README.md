# Consensus AMR: Estimating Central Semantic Graphs via Fréchet Means in Graph Space

Code and data to reproduce the experiments of the paper (Findings of EACL 2027, [citation](#citation)). Given the AMRs of a set of related sentences (a population), the pipeline approximates their Fréchet mean under 1 − SMATCH in two steps: a *semantic mean* in Graph Space, estimated with mean-AAC, whose entries are the proportions of inputs supporting each aligned concept and relation; then a regularized support-maximization decoder that turns it into a well-formed AMR, the *consensus AMR*. The consensus is compared with the selection baselines (text, SMATCH and S2MATCH medoids). Three corpora are used, one population per text:

| Corpus | Populations | Node cap | Mean-AAC starts | Data |
| ------ | ----------- | -------- | --------------- | ---- |
| *The Little Prince* | 27 chapters | 15 | 20 | `data/petit_prince/`, included |
| Bio AMR | 44 PubMed papers | 34 | 9 | `data/bio/`, included |
| AMR 3.0 proxy reports | 367 newswire articles | 24 | 20 | `data/proxy/`, prepared from LDC2020T02, not included |

## Install

```bash
pip install -r requirements.txt
```

S2MATCH downloads GloVe vectors (about 130 MB) and the text-medoid baseline a
SentenceBERT encoder (about 90 MB) on first use.

## Run

```bash
# The Little Prince (defaults: data/petit_prince, results/, cap 15, 20 starts)
python main.py search --jobs 8   # a Fréchet mean per chapter, then sweep τ
python main.py aggregate         # turn the search into the reported tables
python main.py all --jobs 8      # both

# Bio AMR
python main.py all --data data/bio --out results_bio --no-drop-heading \
    --max-nodes 34 --restarts 9 --jobs 8

# AMR 3.0 proxy reports: copy amr-release-3.0-amrs-proxy.txt of LDC2020T02 to
# data/ldc_raw/proxy.txt, split it into one file per article, then run
python main.py prepare --corpora proxy
python main.py all --data data/proxy --out results_proxy --no-drop-heading \
    --max-nodes 24 --jobs 8
```

On the Little Prince `search` takes about ninety minutes with `--jobs 8`. Semantic means are cached under `<out>/means/`, keyed on the population's content, so a second search over other values of τ only redoes the sweep. `aggregate` runs in seconds to a minute.

`--threshold-range` and `--threshold-step` move the grid of τ, `--restarts` the number of mean-AAC runs per population, `--max-nodes` the node cap above which graphs are dropped, `--no-drop-heading` reads population files that have no heading graph (the Little Prince chapters open on "Chapter N .", which is dropped by default), and `--min-graphs` skips populations left with fewer graphs. Every setting used is recorded in `<out>/config.json`. `--jobs` spreads the mean-AAC runs themselves over the processes, not only the populations; the means are the same bit for bit whatever the number of jobs.

## Outputs

| File                              | Content                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------- |
| `<out>/experiment_scores.csv`     | one row per experiment × metric × baseline: the two scores, the gain, its tests     |
| `<out>/selected_params.csv`       | the τ each experiment and metric applies to each population                        |
| `<out>/consensus_graph_smatch.csv`  | the consensus AMRs of *main*, SMATCH-selected                                    |
| `<out>/consensus_graph_s2match.csv` | the same, S2MATCH-selected                                                       |

`<out>` is `results/` (Little Prince), `results_bio/` and `results_proxy/`. `search` also writes `<out>/scores.csv.gz` and `<out>/graphs.csv.gz`, every score and every graph produced by the sweep, and caches the semantic means in `<out>/means/`; these are not versioned. The consensus graphs of `results_proxy/` derive from AMR 3.0 and are not versioned either.

The three experiments differ only in how τ is chosen:

- **main** — tuned on the evaluated population (*per-population τ* in the paper);
- **global_in_sample** — a single τ for the whole corpus (not reported in the paper);
- **loco** — leave-one-population-out (*LOPO* in the paper): chosen on the other populations, read on the held-out one.

A population with no consensus at the τ an experiment selects would be left out and listed in `<out>/populations_without_consensus.csv`; none is on the three corpora.

Columns of `experiment_scores.csv`. The unit of every test is the population, and the gain is the consensus score minus the baseline score (Eq. 12):

| Column | Content |
| ------ | ------- |
| `n_chapters` | number of populations |
| `score_meanaac`, `score_baseline`, `gain`, `median_gain` | mean scores, mean and median gain |
| `n_better`, `n_worse`, `n_tied` | populations where the consensus scores higher, lower, the same |
| `p_ttest`, `p_sign` | one-sided paired t-test and sign test (consensus better) |
| `n_sig_better`, `n_sig_worse` | populations where a one-sided paired t-test over the inputs is significant at 0.05 |
| `p_perm`, `p_perm_greater` | paired permutation (sign-flip) test on the mean gain, two-sided and one-sided: exact enumeration of the 2^n sign vectors up to 20 populations, 200,000 random sign vectors beyond (seed 0, p-value (b + 1)/(B + 1)) |
| `ci95_low`, `ci95_high` | 95 % percentile bootstrap confidence interval of the mean gain, 20,000 resamples of the populations (seed 0) |
| `p_wilcoxon`, `p_wilcoxon_greater` | Wilcoxon signed-rank test, two-sided and one-sided, zero gains discarded |
| `p_sign_2s`, `p_ttest_2s` | two-sided sign test and paired t-test |

The tests are in `consensus/stats.py`; all randomness is seeded, so the tables are reproducible bit for bit.

## Further experiments

The scripts of `experiments/` read the sweeps above and write to `outputs/<name>/`.

| Script | Content | Output |
| ------ | ------- | ------ |
| `prepare_bio.py` | Bio AMR populations from the release file: one per PubMed paper, `:xref` grounding layer and section titles removed | `data/bio/`, `outputs/prepare_bio/` |
| `corpus_stats.py` | preprocessing statistics of the three corpora: AMRs removed at each step, kept under the cap, ranges of n, N, C, R | `outputs/corpus_stats/` |
| `check_constraints.py results results_bio results_proxy` | number of selected consensus graphs that satisfy each structural condition (C1)–(C7) of the paper | `outputs/constraints/summary.csv` |
| `coherence.py --jobs 8` | mean pairwise SMATCH within the chapters of the Little Prince against across chapters, with its paired tests | `outputs/coherence/` |
| `describe.py results results_bio results_proxy` | what the SMATCH-selected consensus graphs contain: acyclicity, re-entrancies, duplicated concepts, whether one input contains all their concepts | `outputs/describe/` |
| `medoid_advantage.py` | decomposition of the gain over the medoid (Eq. 18): gain on the inputs other than the medoid (`fair_term`) minus the medoid's self-match advantage (`advantage_exact`) | `outputs/medoid_advantage/` |
| `stability.py --jobs 8` | all 20 mean-AAC starts of every chapter, their energy and the consensus each gives; checks that the lowest-energy start is the cached mean | `outputs/stability/runs.csv`, `check.txt` |
| `stability_analysis.py` | whether the gain depends on the start kept: agreement between starts, gain per picker, gain against the number of starts | `outputs/stability/` |
| `synthetic_recovery.py --jobs 2` | recovery of a hidden graph: 30 seed AMRs of the Little Prince, copied n ∈ {5, 10, 20} times with noise rate ρ ∈ {0.1, 0.2, 0.3}; consensus and medoid scored against the seed | `outputs/synthetic/` |
| `timing.py --jobs 20` | running time of mean-AAC, the decoder and the baselines, against n (largest chapter) and against the node cap (largest proxy article) | `outputs/timing/timing.csv` |

`stability.py`, `synthetic_recovery.py` and `timing.py` take hours; their outputs are included. `timing.py` should run with `OMP_NUM_THREADS=1`, and its times depend on the machine.

## Where each result of the paper comes from

| Paper | File | What to read |
| ----- | ---- | ------------ |
| Table 1, Section 7.1 | `results/experiment_scores.csv` | rows `main` and `loco`: `score_meanaac`, `score_baseline`, `gain`, `ci95_low`, `ci95_high`, `p_perm`, `n_better` |
| LOPO thresholds, Section 7.1 | `results/selected_params.csv` | τ per experiment, metric and chapter |
| Table 2 | `results/`, `results_bio/`, `results_proxy/` → `experiment_scores.csv` | rows whose baseline is the medoid of the same metric |
| Section 7.2, Figures 2 and 3, Appendix C | `results/consensus_graph_smatch.csv`, `outputs/describe/` | consensus AMRs and what they contain |
| Section 6.1 (within / across chapters) | `outputs/coherence/summary.csv` | `within`, `across`, `gain`, `ci95_*` |
| (C1)–(C7) on the outputs, Sections 7.1 and 7.4 | `outputs/constraints/summary.csv` | consensus graphs satisfying each condition |
| Section 7.3, Table 3 | `outputs/synthetic/summary.csv` | `consensus_main`, `medoid_smatch`, `exact_consensus`, `exact_medoid` (shares of 30 trials) |
| Section 7.4, Table 4 | `outputs/medoid_advantage/summary.csv` | `fair_term` (other inputs) and `advantage_exact` (self-match), protocol `main`, SMATCH |
| Appendix A.5 | `outputs/timing/timing.csv` | `mean_starts_cpu` (mean-AAC, all starts) and `medoid_smatch` |
| Appendix B.1 | `outputs/corpus_stats/summary.json` | preprocessing counts |
| Appendix B.2 (variability) | `outputs/stability/summary.csv` | agreement and score spread between starts |

## Code

| Module                   | Content                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| `consensus/corpus.py`    | AMR files to tensor encodings, and back to Penman                            |
| `consensus/align.py`     | Frank-Wolfe alignment, the quotient distance                                 |
| `consensus/mean.py`      | mean-AAC                                                                     |
| `consensus/project.py`   | the regularized support-maximization decoder: connectivity, role determinism, core-role licensing |
| `consensus/metrics.py`   | SMATCH and S2MATCH, via smatchpp                                             |
| `consensus/baselines.py` | the three selection baselines                                                |
| `consensus/search.py`    | the means, one restart per process; the sweep over τ: one study per population, one over the corpus |
| `consensus/aggregate.py` | experimental results                                                         |
| `consensus/stats.py`     | permutation, Wilcoxon and sign tests, percentile bootstrap                   |
| `consensus/ldc.py`       | AMR 3.0 proxy release file to one population file per article               |

## Data

Preprocessing keeps instantiated concepts only and drops the graphs above the node cap; the counts below are those of `outputs/corpus_stats/summary.json`.

- **The Little Prince** — `data/petit_prince/` holds the 27 chapters of the AMR-annotated *Little Prince* (AMR release 3.0, [https://amr.isi.edu](https://amr.isi.edu)), one file per chapter. Each chapter's heading graph is dropped; 1,465 of the 1,562 graphs are kept.
- **Bio AMR** — Bio AMR v0.8 (cancer-related PubMed articles, CC BY 4.0), from the figshare mirror of the ISI release (article 3206062, file `amr-bio.txt.grounded.txt`), stored as `data/open_raw/amr-bio-v0.8.grounded.txt`. `experiments/prepare_bio.py` writes `data/bio/`: one population per PubMed paper (the BEL, Chicago and other non-PubMed sentences are left out), the `:xref` grounding layer of the mirror removed, section titles removed. 3,669 of the 3,836 body graphs are kept under the cap of 34 nodes. Both the release file and the populations are redistributed under CC BY 4.0.
- **AMR 3.0 proxy reports** — AMR 3.0 (LDC2020T02) is distributed by the Linguistic Data Consortium under license and is not redistributed here. With the release file `amr-release-3.0-amrs-proxy.txt` (8,252 AMRs) copied to `data/ldc_raw/proxy.txt`, `python main.py prepare --corpora proxy` writes `data/proxy/`, one population per article holding its body sentences (`::snt-type` `body` and `body subordinate`), and lists the date, country, topic and summary sentences it leaves out in `data/proxy_removed.csv`. 6,284 of the 6,595 body graphs are kept under the cap of 24 nodes. `.gitignore` excludes everything derived from it that holds graphs or sentences.

## Citation

```bibtex
@inproceedings{chappon-etal-2027-consensus,
  title     = {Consensus {AMR}: Estimating Central Semantic Graphs via {F}r{\'e}chet Means in Graph Space},
  author    = {Chappon, Edouard and Guille, Adrien and Cugliari Duhalde, Jairo},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027},
  publisher = {Association for Computational Linguistics}
}
```
