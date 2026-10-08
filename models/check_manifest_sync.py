#!/usr/bin/env python3
"""Assert that every place naming the released checkpoints agrees.

The released model set is written down in two forms, and both drift.

Release *filenames* (what gets uploaded and downloaded):

    fetch_weights.sh    the WEIGHTS array actually downloaded
    ZENODO_WEIGHTS.md   the upload manifest table
    README.md           the release table and the SHA256SUMS block

Model *ids* (which trained model each file is):

    README.md           the release table's hash column
    ZENODO_WEIGHTS.md   the upload manifest's id column
    ../data/MANIFEST.md the generated "Model ids" table

A checkpoint added to one and not the others has shipped, or nearly shipped, a
deposition without a model the manuscript reports three times (2026-08-20,
2026-08-26, 2026-09-09). This script makes that drift a hard failure instead of
something noticed by reading.

Note that `data/MANIFEST.md` is **generated** by `tools_build_deposition.py`
(which lives in the working tree, not in this repository). If the id check fails
on that file alone, re-run the builder; if it still fails, the builder's own
hard-coded table is what is stale.

Usage:
    ./check_manifest_sync.py          # exit 0 if every source agrees
    ./check_manifest_sync.py -v       # also print the agreed sets

Exit codes:
    0  all sources agree, on filenames and on ids
    1  they disagree (the differences are printed)
    2  a source could not be parsed at all -- treat as unknown, not as pass
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FETCH = HERE / "fetch_weights.sh"
ZENODO = HERE / "ZENODO_WEIGHTS.md"
README = HERE / "README.md"

# A release filename: word characters, dots and dashes, ending in .pt
PT = re.compile(r"[A-Za-z0-9_.\-]+\.pt")

# Checkpoint files that live inside the working tree rather than being release
# artefacts. These appear in provenance columns ("f16b51e6/ckpts/epoch_200.pt")
# and must not be mistaken for files to upload.
WORKING_TREE_NAMES = {"best.pt", "last.pt"}
EPOCH_PT = re.compile(r"^epoch_\d+\.pt$")


def _release_names(names: set[str]) -> set[str]:
    """Drop working-tree checkpoint names, keeping only release artefacts."""
    return {
        n
        for n in names
        if n not in WORKING_TREE_NAMES and not EPOCH_PT.match(n)
    }


def from_fetch_script(text: str) -> set[str]:
    """Parse the WEIGHTS=( ... ) array -- the list actually downloaded."""
    m = re.search(r"^WEIGHTS=\(\s*(.*?)^\)", text, re.S | re.M)
    if not m:
        raise ValueError(
            "no WEIGHTS=( ... ) array found in fetch_weights.sh; if the download "
            "list was restructured, update this parser deliberately rather than "
            "loosening it"
        )
    body = re.sub(r"#.*", "", m.group(1))
    return _release_names(set(PT.findall(body)))


def from_zenodo_manifest(text: str) -> set[str]:
    """Parse the first column of the upload-manifest table rows."""
    names: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        first = line.split("|")[1] if line.count("|") >= 2 else ""
        found = PT.findall(first)
        names.update(found)
    out = _release_names(names)
    if not out:
        raise ValueError("no .pt filenames found in any table's first column")
    return out


def from_readme_table(text: str) -> set[str]:
    """Parse the release-name column of the README release table."""
    names: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        first = line.split("|")[1] if line.count("|") >= 2 else ""
        names.update(PT.findall(first))
    out = _release_names(names)
    if not out:
        raise ValueError("no .pt filenames found in the release table")
    return out


def from_readme_shasums(text: str) -> set[str]:
    """Parse the SHA256SUMS fenced block, which must cover every released file."""
    m = re.search(r"```\s*\n#\s*sha256sum\s+\*\.pt.*?\n(.*?)```", text, re.S)
    if not m:
        raise ValueError("no '# sha256sum *.pt' fenced block found")
    return _release_names(set(PT.findall(m.group(1))))


DATA_MANIFEST = HERE.parent / "data" / "MANIFEST.md"

# A model id: exactly eight lowercase hex characters, optionally backticked, and
# filling its whole table cell. The full-cell requirement is what keeps provenance
# cells such as `f16b51e6/ckpts/epoch_200.pt` out of the id set.
ID_CELL = re.compile(r"^\s*`?([0-9a-f]{8})`?\s*$")


def _ids_in_column(text: str, column: int, section: str | None = None) -> set[str]:
    """Collect model ids from one column of the markdown tables in `text`.

    `column` is 1-based over the cells between pipes. If `section` is given, only
    lines under that `## heading` (up to the next `## `) are considered.
    """
    if section is not None:
        m = re.search(
            rf"^##\s+{re.escape(section)}\s*$(.*?)(?=^##\s|\Z)", text, re.S | re.M
        )
        if not m:
            raise ValueError(f"no '## {section}' section found")
        text = m.group(1)

    ids: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = line.split("|")[1:-1] if line.rstrip().endswith("|") else line.split("|")[1:]
        if len(cells) < column:
            continue
        hit = ID_CELL.match(cells[column - 1])
        if hit:
            ids.add(hit.group(1))
    return ids


def _check(kind: str, sources: dict[str, set[str]]) -> tuple[bool, set[str]]:
    """Report whether every source agrees; returns (ok, union)."""
    union = set().union(*sources.values()) if sources else set()
    disagree = {k: v for k, v in sources.items() if v != union}
    if not disagree:
        return True, union

    print(f"{kind.upper()} OUT OF SYNC -- the released model set is not agreed.\n")
    print(f"union of all sources: {len(union)} {kind}")
    for label, names in sources.items():
        missing = sorted(union - names)
        status = "ok" if not missing else f"MISSING {len(missing)}"
        print(f"\n  {label}: {len(names)} {kind} [{status}]")
        for n in missing:
            print(f"      absent: {n}")
    return False, union


def main() -> int:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    sources: dict[str, set[str]] = {}
    for label, path, parser in [
        ("fetch_weights.sh (WEIGHTS array)", FETCH, from_fetch_script),
        ("ZENODO_WEIGHTS.md (upload manifest)", ZENODO, from_zenodo_manifest),
        ("README.md (release table)", README, from_readme_table),
        ("README.md (SHA256SUMS block)", README, from_readme_shasums),
    ]:
        if not path.exists():
            print(f"CANNOT DETERMINE: {path.name} does not exist", file=sys.stderr)
            return 2
        try:
            sources[label] = parser(path.read_text())
        except ValueError as exc:
            print(f"CANNOT DETERMINE: {label}: {exc}", file=sys.stderr)
            return 2

    files_ok, file_union = _check("files", sources)

    # --- model ids -------------------------------------------------------
    id_sources: dict[str, set[str]] = {}
    try:
        id_sources["README.md (release table, hash column)"] = _ids_in_column(
            README.read_text(), column=2
        )
        id_sources["ZENODO_WEIGHTS.md (manifest, id column)"] = _ids_in_column(
            ZENODO.read_text(), column=2
        )
    except ValueError as exc:
        print(f"CANNOT DETERMINE: model ids: {exc}", file=sys.stderr)
        return 2

    if DATA_MANIFEST.exists():
        try:
            id_sources["data/MANIFEST.md (generated Model ids table)"] = _ids_in_column(
                DATA_MANIFEST.read_text(), column=1, section="Model ids"
            )
        except ValueError as exc:
            print(f"CANNOT DETERMINE: data/MANIFEST.md: {exc}", file=sys.stderr)
            return 2
    else:
        print(
            "note: data/MANIFEST.md absent, so the generated id table was not "
            "checked (run tools_build_deposition.py to produce it)."
        )

    if any(not v for v in id_sources.values()):
        empty = [k for k, v in id_sources.items() if not v]
        print(f"CANNOT DETERMINE: no ids parsed from: {empty}", file=sys.stderr)
        return 2

    if files_ok:
        print(f"OK: all {len(sources)} sources name the same {len(file_union)} checkpoints.")
        if verbose:
            for n in sorted(file_union):
                print(f"    {n}")
        print()

    ids_ok, id_union = _check("ids", id_sources)
    if ids_ok:
        print(f"OK: all {len(id_sources)} sources name the same {len(id_union)} model ids.")
        if verbose:
            for n in sorted(id_union):
                print(f"    {n}")

    if not (files_ok and ids_ok):
        print(
            "\nFix every source before uploading. A deposition built from an "
            "out-of-sync manifest omits models the manuscript reports -- this has "
            "happened three times (see the corrections in README.md)."
        )
        return 1

    if len(file_union) != len(id_union):
        print(
            f"\nMANIFEST OUT OF SYNC: {len(file_union)} release files but "
            f"{len(id_union)} model ids. Each released file is exactly one trained "
            "model, so these counts must match."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
