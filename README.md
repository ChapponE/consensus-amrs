# Consensus AMR

Code and data for *Consensus AMR: Estimating Central Semantic Graphs via Fréchet Means in Graph Space* (Chappon, Guille and Cugliari Duhalde, Findings of EACL 2027). The method is in `consensus/`: AMR encoding, Frank–Wolfe alignment, mean-AAC, the decoder, the baselines and the metrics.

## Install

```bash
pip install -r requirements.txt
```

S2MATCH downloads GloVe vectors (about 130 MB) and the text medoid a Sentence-BERT encoder (about 90 MB) on first use.

## Reproduce

```bash
# Tables 1 and 2, Figures 2 and 3 (The Little Prince: about 1.5 h with 8 jobs)
python main.py all --jobs 8
python main.py all --data data/bio --out results_bio --no-drop-heading --max-nodes 34 --restarts 9 --jobs 8
# News: copy amr-release-3.0-amrs-proxy.txt of AMR 3.0 (LDC2020T02) to data/ldc_raw/proxy.txt, then
python main.py prepare --corpora proxy
python main.py all --data data/proxy --out results_proxy --no-drop-heading --max-nodes 24 --jobs 8

# Other results of Sections 6 and 7, written to outputs/<name>/
python experiments/coherence.py --jobs 8
python experiments/corpus_stats.py
python experiments/check_constraints.py results results_bio results_proxy
python experiments/describe.py results results_bio results_proxy
python experiments/medoid_advantage.py
python experiments/synthetic_recovery.py --jobs 2   # several hours
```

All randomness is seeded: the tables come out identical.

## Data

- **The Little Prince**: `data/petit_prince/`, one file per chapter, from the AMR release 3.0 of the book ([amr.isi.edu](https://amr.isi.edu)).
- **Bio AMR** v0.8 (Garg et al., 2016), by ISI/USC and the AMR Development Team, released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) on [figshare](https://doi.org/10.6084/m9.figshare.3206062.v4). `data/open_raw/amr-bio-v0.8.grounded.txt` is the release file `amr-bio.txt.grounded.txt`, unchanged. `data/bio/` is our modified version, built from it by `experiments/prepare_bio.py`: one file per PubMed article, with the `:xref` grounding layer and the section titles removed.
- **News**: AMR 3.0 is licensed by the LDC and is not included; `main.py prepare` builds `data/proxy/` from the release file.

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
