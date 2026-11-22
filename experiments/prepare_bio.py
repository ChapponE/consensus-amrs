"""Bio AMR populations: one file per PubMed paper, written to ``data/bio/``.

**Source.** Bio AMR v0.8 (cancer-related PubMed articles, CC BY 4.0), from the figshare
mirror of the ISI release (article 3206062, file ``amr-bio.txt.grounded.txt``,
6 452 AMRs), stored as ``data/open_raw/amr-bio-v0.8.grounded.txt``.

**Populations.** One population per paper: the ids ``pmid_X_Y.k`` and ``a_pmid_X_Y.k``
(no paper appears under both). The BEL and Chicago sentences are isolated sentences,
not documents, and the small non-PubMed groups (``bio.bmtr``, ``bio.mskcc``…) are not
papers either: both are left out. Two edits:

- the grounding layer the mirror adds (``:xref`` to ``*OR*``/``xref`` nodes) is
  removed, and the graph re-encoded (round trip checked triple by triple);
- section titles are removed: sentences tagged ``<sec-title>``, the paper title
  (it ends with ``(PMID:n)``), and — for papers without markup — sentences that,
  once markup and trailing citations are stripped, end without sentence punctuation
  ("Results", "Figure 3", subsection headings). The three sentences that last rule
  catches but that are not titles are listed in ``NOT_TITLES`` and kept.

The files carry no heading graph, so they are read with ``--no-drop-heading``. The
titles removed are listed in ``outputs/prepare_bio/bio_removed_titles.csv`` and the
population sizes at the node cap in ``outputs/prepare_bio/sizes_bio.csv``.

    python experiments/prepare_bio.py --max-nodes 34
    python main.py all --data data/bio --out results_bio --no-drop-heading \\
        --max-nodes 34 --restarts 9 --jobs 8
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import penman
from penman.models.amr import model as AMR_MODEL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consensus.corpus import _split_blocks, load_chapter  # noqa: E402

BIO_RAW = ROOT / "data" / "open_raw" / "amr-bio-v0.8.grounded.txt"
DATA = ROOT / "data" / "bio"
OUT = ROOT / "outputs" / "prepare_bio"

GROUNDING_CONCEPTS = {"xref", "*OR*"}
NOT_TITLES = {
    "pmid_1563_0473.269",   # "Western blot signals were developed using ... from Amersham" (no final period)
    "pmid_1177_7939.240",   # "HA-Sos, HA-epitope tagged Sos-1" (abbreviation in a figure legend)
    "a_pmid_2514_2146.234",  # legend sentence cut after "*, P"
}
_PAPER_RE = re.compile(r"^(?:a_)?(pmid_\d+_\d+)\.\d+$")
_ID_RE = re.compile(r"# ::id (\S+)")
_SNT_RE = re.compile(r"# ::snt (.*)")


def _title_reason(snt: str, amr_id: str) -> str | None:
    if amr_id in NOT_TITLES:
        return None
    if "<sec-title" in snt:
        return "sec-title tag"
    if re.search(r"\(PMID:\d+\)\s*$", snt):
        return "paper title"
    body = re.sub(r"(\s*<xref[^>]*>.*?</xref>\s*[,;–-]?\s*)+$", "", snt)
    body = re.sub(r"<[^>]+>", "", body).strip()
    body = re.sub(r"(\s*[\[(][^\])]*[\])])+\s*$", "", body)
    if re.search(r"[.!?:;]['\"”]*$", body) or "<xref" in snt or len(body.split()) > 25:
        return None
    return "no final punctuation"


def _without_grounding(block: str) -> str:
    """The block with the mirror's :xref layer removed; other blocks come back verbatim."""
    g = penman.decode(block, model=AMR_MODEL)
    grounding = {s for s, r, t in g.triples if r == ":instance" and t in GROUNDING_CONCEPTS}
    if not grounding:
        return block
    triples = [(s, r, t) for s, r, t in g.triples if s not in grounding and t not in grounding]
    body = penman.encode(penman.Graph(triples, top=g.top), model=AMR_MODEL)
    again = penman.decode(body, model=AMR_MODEL).triples
    if sorted(again) != sorted(triples):
        raise ValueError(f"round trip changed {g.metadata.get('id')}")
    comments = [line for line in block.splitlines() if line.startswith("#")]
    return "\n".join(comments + [body])


def bio_populations(raw: Path = BIO_RAW) -> tuple[OrderedDict[str, list[str]], list[dict]]:
    """The body sentences of every paper, in file order, and the section titles removed."""
    papers: OrderedDict[str, list[str]] = OrderedDict()
    removed = []
    for block in _split_blocks(raw.read_text(encoding="utf-8")):
        match = _ID_RE.search(block)
        paper = _PAPER_RE.match(match.group(1)) if match else None
        if paper is None:
            continue
        amr_id = match.group(1)
        snt = (_SNT_RE.search(block) or [None, ""])[1]
        reason = _title_reason(snt, amr_id)
        if reason:
            removed.append({"id": amr_id, "reason": reason, "snt": re.sub(r"<[^>]+>", "", snt)})
            continue
        papers.setdefault(paper.group(1), []).append(_without_grounding(block))
    return papers, removed


def write(populations: OrderedDict[str, list[str]], directory: Path, out: Path,
          max_nodes: int) -> pd.DataFrame:
    """One file per population, and their sizes before and after the node cap."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    rows = []
    for name, blocks in populations.items():
        path = directory / f"{name}.txt"
        path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        try:
            kept = load_chapter(path, max_nodes=max_nodes, drop_heading=False).nb
        except ValueError:
            kept = 0
        rows.append({"population": name, "graphs": len(blocks), f"kept_n{max_nodes}": kept})
    sizes = pd.DataFrame(rows)
    sizes.to_csv(out / "sizes_bio.csv", index=False)
    print(f"bio: {len(sizes)} populations, {sizes['graphs'].sum()} graphs, "
          f"{sizes[f'kept_n{max_nodes}'].sum()} at <= {max_nodes} nodes, written to {directory}")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=BIO_RAW, help="the Bio AMR v0.8 release file")
    parser.add_argument("--data", type=Path, default=DATA, help="directory receiving the populations")
    parser.add_argument("--out", type=Path, default=OUT, help="directory receiving the two tables")
    parser.add_argument("--max-nodes", type=int, default=34,
                        help="node cap the sizes table is computed at")
    args = parser.parse_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    args.out = resolve(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    papers, removed = bio_populations(resolve(args.raw))
    pd.DataFrame(removed).to_csv(args.out / "bio_removed_titles.csv", index=False)
    print(f"bio: {len(removed)} section titles removed "
          f"({pd.Series([r['reason'] for r in removed]).value_counts().to_dict()})")
    write(papers, resolve(args.data), args.out, args.max_nodes)


if __name__ == "__main__":
    main()
