"""Preprocessing statistics of the three corpora, read from the data files.

For every corpus: how many AMRs the source holds, what each preparation step removes,
how many body AMRs the populations hold before the node cap, how many the cap keeps,
and, over the kept graphs, the mean number of nodes and the ranges over populations of
the population size n, the padded size N (largest kept graph), the concept vocabulary
C and the role vocabulary R. Kept graphs and nodes are counted as ``load_chapter``
counts them (a graph whose root ``prune_graph`` removes is dropped before the cap), so the numbers are those the pipeline runs on.

- **petit_prince** — ``data/petit_prince/``, heading graph dropped, cap 15;
- **bio** — ``data/open_raw/amr-bio-v0.8.grounded.txt`` read through
  ``prepare_bio.bio_populations``, then ``data/bio/``, cap 34;
- **proxy** — ``data/ldc_raw/proxy.txt`` (AMR 3.0, not redistributed) and the
  ``data/proxy_removed.csv`` written by ``main.py prepare``, then ``data/proxy/``, cap 24.
  Skipped when the LDC files are absent.

    python experiments/corpus_stats.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from consensus.corpus import (_split_blocks, load_chapter, natural_key,  # noqa: E402
                              parse_amr_file, prune_graph)
import prepare_bio  # noqa: E402

OUT = ROOT / "outputs" / "corpus_stats"
CORPORA = {
    "petit_prince": dict(data=ROOT / "data" / "petit_prince", max_nodes=15, drop_heading=True),
    "bio": dict(data=ROOT / "data" / "bio", max_nodes=34, drop_heading=False),
    "proxy": dict(data=ROOT / "data" / "proxy", max_nodes=24, drop_heading=False),
}
_ID_RE = re.compile(r"::id (\S+)")


def populations(data: Path, max_nodes: int, drop_heading: bool) -> pd.DataFrame:
    """One row per population: AMRs in the file, before and after the cap, and N, C, R."""
    rows = []
    for path in sorted(data.glob("*.txt"), key=lambda p: natural_key(p.stem)):
        graphs = parse_amr_file(path)
        pruned = [g for g in (prune_graph(g) for g in graphs) if g is not None]
        above = sum(len(g.variables()) > max_nodes for g in pruned)
        chapter = load_chapter(path, max_nodes=max_nodes, drop_heading=drop_heading)
        nodes = (chapter.nf.sum(axis=2) > 0).sum(axis=1)
        heading = int(drop_heading)
        rows.append(dict(population=path.stem, amrs_in_file=len(graphs),
                         body_amrs=len(graphs) - heading,
                         root_pruned=len(graphs) - len(pruned), above_cap=above,
                         undecodable=len(pruned) - above - heading - chapter.nb, kept=chapter.nb,
                         nodes_sum=int(nodes.sum()), N=chapter.nf.shape[1],
                         C=len(chapter.concepts), R=len(chapter.roles)))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, max_nodes: int) -> dict:
    body, kept = int(df["body_amrs"].sum()), int(df["kept"].sum())
    return dict(
        populations=len(df), max_nodes=max_nodes,
        amrs_in_files=int(df["amrs_in_file"].sum()),
        body_amrs_before_cap=body,
        dropped_root_pruned=int(df["root_pruned"].sum()),
        dropped_above_cap=int(df["above_cap"].sum()),
        dropped_undecodable=int(df["undecodable"].sum()),
        kept=kept, kept_share_percent=round(100 * kept / body, 2),
        kept_share_of_unpruned_percent=round(100 * kept / (body - int(df["root_pruned"].sum())), 2),
        nodes_mean_per_kept_graph=round(df["nodes_sum"].sum() / kept, 3),
        n_range=[int(df["kept"].min()), int(df["kept"].max())],
        n_median=float(df["kept"].median()),
        N_range=[int(df["N"].min()), int(df["N"].max())],
        C_range=[int(df["C"].min()), int(df["C"].max())],
        R_range=[int(df["R"].min()), int(df["R"].max())],
    )


def bio_source() -> dict:
    """What the Bio AMR release holds, and what the preparation removes from it."""
    blocks = [b for b in _split_blocks(prepare_bio.BIO_RAW.read_text(encoding="utf-8"))
              if _ID_RE.search(b)]
    groups = Counter()
    for block in blocks:
        amr_id = _ID_RE.search(block).group(1)
        if prepare_bio._PAPER_RE.match(amr_id):
            groups["pubmed_papers"] += 1
        elif "bel" in amr_id.lower():
            groups["bel"] += 1
        elif "chicago" in amr_id.lower():
            groups["chicago"] += 1
        else:
            groups["other_non_pubmed"] += 1
    papers, removed = prepare_bio.bio_populations()
    body = [b for blocks_ in papers.values() for b in blocks_]
    grounded = sum(prepare_bio._without_grounding(b) != b
                   for b in (b for b in blocks if prepare_bio._PAPER_RE.match(_ID_RE.search(b).group(1))))
    titles = {r["id"] for r in removed}
    grounded_body = sum(prepare_bio._without_grounding(b) != b for b in blocks
                        if prepare_bio._PAPER_RE.match(_ID_RE.search(b).group(1))
                        and _ID_RE.search(b).group(1) not in titles)
    other_ids = Counter(re.sub(r"\.\d+$", "", _ID_RE.search(b).group(1)) for b in blocks
                        if not prepare_bio._PAPER_RE.match(_ID_RE.search(b).group(1))
                        and "bel" not in _ID_RE.search(b).group(1).lower()
                        and "chicago" not in _ID_RE.search(b).group(1).lower())
    return dict(
        amrs_in_source=len(blocks),
        by_group=dict(groups),
        left_out_not_pubmed=len(blocks) - groups["pubmed_papers"],
        non_pubmed_other_groups=len(other_ids),
        papers=len(papers),
        pubmed_amrs=groups["pubmed_papers"],
        section_titles_removed=len(removed),
        section_titles_by_reason=dict(Counter(r["reason"] for r in removed)),
        body_amrs=len(body),
        pubmed_graphs_with_xref_layer_removed=int(grounded),
        body_graphs_with_xref_layer_removed=int(grounded_body),
    )


def proxy_source() -> dict | None:
    raw = ROOT / "data" / "ldc_raw" / "proxy.txt"
    removed = ROOT / "data" / "proxy_removed.csv"
    if not raw.exists() or not removed.exists():
        return None
    blocks = [b for b in _split_blocks(raw.read_text(encoding="utf-8")) if _ID_RE.search(b)]
    reasons = pd.read_csv(removed, keep_default_na=False)["reason"].value_counts()
    return dict(amrs_in_source=len(blocks),
                documents=len({_ID_RE.search(b).group(1).rsplit(".", 1)[0] for b in blocks}),
                removed_total=int(reasons.sum()),
                removed_by_snt_type={str(k): int(v) for k, v in reasons.items()})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, spec in CORPORA.items():
        if not any(spec["data"].glob("*.txt")):
            print(f"{name}: no population in {spec['data']}, skipped")
            continue
        df = populations(spec["data"], spec["max_nodes"], spec["drop_heading"])
        df.to_csv(OUT / f"per_population_{name}.csv", index=False)
        summary[name] = summarise(df, spec["max_nodes"])
        source = {"bio": bio_source, "proxy": proxy_source}.get(name)
        if source is not None and (found := source()) is not None:
            summary[name]["source"] = found
        print(name, json.dumps(summary[name], indent=1), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
