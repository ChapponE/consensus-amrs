"""Populations from AMR 3.0 (LDC2020T02), one file per document: the proxy reports.

The Little Prince comes split by chapter; AMR 3.0 comes as one release file per
genre, which this module splits by document — the id of every AMR is
``<document>.<sentence>``. What a population keeps mirrors what a chapter keeps:
the sentences of the text, without the heading material around them.

- **proxy** (AFP and other newswire, the PROXY reports): every sentence carries a
  ``::snt-type``. The population is the article body (``body`` and
  ``body subordinate``); the ``date``, ``country`` and ``topic`` fields are
  metadata. The human ``summary`` sentences are held out and written apart, to
  ``proxy_summaries/``.

The files are written verbatim, block by block, so ``load_chapter`` preprocesses
them exactly as it does the chapters, with ``drop_heading=False``. Every AMR left
out is listed, with the reason, in ``<corpus>_removed.csv`` next to the populations.
"""
from __future__ import annotations

import csv
import re
from collections import OrderedDict
from pathlib import Path

from .corpus import _split_blocks

CORPORA = ("proxy",)
BODY_TYPES = {"body", "body subordinate"}
_ID_RE = re.compile(r"::id (\S+)")
_SNT_RE = re.compile(r"::snt (.*)")
_TYPE_RE = re.compile(r"::snt-type (.+?)(?= ::|$)", re.MULTILINE)


def _documents(path: Path) -> OrderedDict[str, list[str]]:
    """AMR blocks of a release file grouped by document, in file order."""
    documents: OrderedDict[str, list[str]] = OrderedDict()
    for block in _split_blocks(path.read_text(encoding="utf-8")):
        match = _ID_RE.search(block)
        if match is None:  # the release header
            continue
        documents.setdefault(match.group(1).rsplit(".", 1)[0], []).append(block)
    return documents


def _snt_type(block: str) -> str | None:
    match = _TYPE_RE.search(block)
    return match.group(1).strip() if match else None


def split(corpus: str, release_file: Path, out_dir: Path) -> dict[str, int]:
    """Write one population file per document of ``corpus``; return their sizes in AMRs.

    The AMRs left out are listed in ``<out_dir>/<corpus>_removed.csv``.
    """
    if corpus not in CORPORA:
        raise ValueError(f"Unknown corpus {corpus!r}; expected one of {CORPORA}.")
    population_dir = Path(out_dir) / corpus
    population_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = Path(out_dir) / f"{corpus}_summaries"

    sizes, removed_rows = {}, []
    for document, blocks in _documents(Path(release_file)).items():
        removed = {i: _snt_type(b) for i, b in enumerate(blocks) if _snt_type(b) not in BODY_TYPES}
        summaries = [b for b in blocks if _snt_type(b) == "summary"]
        if summaries:
            summary_dir.mkdir(parents=True, exist_ok=True)
            (summary_dir / f"{document}.txt").write_text("\n\n".join(summaries) + "\n",
                                                        encoding="utf-8")
        for i, reason in sorted(removed.items()):
            snt = _SNT_RE.search(blocks[i])
            removed_rows.append({"document": document, "id": _ID_RE.search(blocks[i]).group(1),
                                 "reason": reason, "snt": snt.group(1).strip() if snt else ""})
        kept = [b for i, b in enumerate(blocks) if i not in removed]
        if kept:
            (population_dir / f"{document}.txt").write_text("\n\n".join(kept) + "\n",
                                                            encoding="utf-8")
            sizes[document] = len(kept)

    if removed_rows:
        with open(Path(out_dir) / f"{corpus}_removed.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["document", "id", "reason", "snt"])
            writer.writeheader()
            writer.writerows(removed_rows)
    return sizes


def run(raw_dir: Path, out_dir: Path, corpora=CORPORA) -> None:
    """Split every release file ``<corpus>.txt`` of ``corpora`` found in ``raw_dir``."""
    for corpus in corpora:
        release_file = Path(raw_dir) / f"{corpus}.txt"
        if not release_file.exists():
            print(f"  {corpus}: no {release_file}, skipped")
            continue
        sizes = split(corpus, release_file, out_dir)
        print(f"  {corpus}: {len(sizes)} documents, {sum(sizes.values())} AMRs "
              f"written to {Path(out_dir) / corpus}")
