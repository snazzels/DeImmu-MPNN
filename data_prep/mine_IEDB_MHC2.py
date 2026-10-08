#!/usr/bin/env python
"""
Mine IEDB for MHC class II immunogenicity training data (WP1).

Produces three peptide sets:
  1. immunogenic  — MHC-II restricted, positive T-cell assay
  2. treg         — tolerance/suppression readout, or Treg markers
  3. human_nonim  — 15-mers from human proteome absent from IEDB immunogenic set

Output directory layout:
  <output>/immunogenic.csv
  <output>/treg.csv
  <output>/human_nonim.csv
  <output>/combined.csv          (all three classes, labelled)
  <output>/stats.txt
"""

import argparse
import gzip
import io
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IEDB_TCELL_URL = (
    "https://www.iedb.org/downloader.php?file_name=doc/tcell_full_v3.zip"
)

UNIPROT_HUMAN_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?format=fasta&query=reviewed:true+AND+organism_id:9606"
)

PEPTIDE_LENGTH = 15

# MHC-II genes to keep (netMHCIIpan-relevant)
MHC2_GENES = {"DRB1", "DRB3", "DRB4", "DRB5", "DQA1", "DQB1", "DPA1", "DPB1"}

# Assay outcome keywords that indicate positive T-cell activation
POSITIVE_QUALITATIVE = {"Positive", "Positive-High", "Positive-Intermediate",
                         "Positive-Low"}

# Canonical effector T-cell readouts — used to define the immunogenic set.
# Using ALL positive assays would include IL-10+ responses, which overlap
# almost entirely with the Treg set (same peptide, different donor/context).
EFFECTOR_RESPONSES = {
    "IFNg release", "proliferation", "cytotoxicity", "TNFa release", "TNF release",
    "activation", "degranulation", "granzyme B release", "antibody help",
    "T cell help", "IFNa release", "IFNb release",
}

# IEDB `Assay|Response_measured` values that indicate a tolerogenic / Treg response.
# IL-10 and TGF-β are canonical immunosuppressive cytokines; suppression/tolerance
# are explicit IEDB assay outcome categories.
TREG_RESPONSES = {
    "suppression",
    "tolerance",
    "tolerance after adoptive transfer",
    "IL-10 release",
    "TGFb release",
}

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid_peptide(seq: str, length: int = PEPTIDE_LENGTH) -> bool:
    return len(seq) == length and set(seq).issubset(STANDARD_AA)


def download_with_progress(url: str, desc: str) -> bytes:
    print(f"Downloading {desc} ...")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            buf.write(chunk)
            bar.update(len(chunk))
    buf.seek(0)
    return buf.read()


def load_iedb_tcell(cache_path: Path, iedb_file: Path | None = None) -> pd.DataFrame:
    if cache_path.exists():
        print(f"Using cached IEDB T-cell file: {cache_path}")
        return pd.read_csv(cache_path, low_memory=False, header=[0, 1])

    if iedb_file is None:
        sys.exit(
            "ERROR: IEDB T-cell data not found.\n\n"
            "IEDB requires accepting their terms of service before downloading.\n"
            "Please download 'tcell_full_v3.zip' manually:\n"
            "  1. Go to https://www.iedb.org/database_export_v3.php\n"
            "  2. Accept the terms and download 'T Cell Assay Data (tcell_full_v3.zip)'\n"
            "  3. Re-run with: --iedb_file /path/to/tcell_full_v3.zip\n"
        )

    iedb_file = Path(iedb_file)
    if iedb_file.suffix == ".zip":
        with zipfile.ZipFile(iedb_file) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, low_memory=False, header=[0, 1])
    else:
        df = pd.read_csv(iedb_file, low_memory=False, header=[0, 1])

    df.to_csv(cache_path, index=False)
    print(f"Cached to {cache_path}")
    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse IEDB's two-row header into single strings."""
    df.columns = [
        f"{a}|{b}".strip("|").replace(" ", "_")
        if str(b) not in ("", "nan", "Unnamed") and str(a) not in ("", "nan", "Unnamed")
        else (str(a) if str(b) in ("", "nan") else str(b)).replace(" ", "_")
        for a, b in df.columns
    ]
    return df


def extract_mhc2_gene(allele_str: str) -> str | None:
    """Return the MHC gene name (e.g. 'DRB1') from an allele string, or None."""
    if not isinstance(allele_str, str):
        return None
    for gene in MHC2_GENES:
        if gene in allele_str:
            return gene
    return None


def slide_15mers(sequence: str) -> list[str]:
    return [
        sequence[i: i + PEPTIDE_LENGTH]
        for i in range(len(sequence) - PEPTIDE_LENGTH + 1)
        if is_valid_peptide(sequence[i: i + PEPTIDE_LENGTH])
    ]


def load_human_proteome(cache_path: Path) -> list[str]:
    """Returns list of canonical human protein sequences (SwissProt)."""
    if cache_path.exists():
        print(f"Using cached human proteome: {cache_path}")
        with open(cache_path) as f:
            seqs = [l.strip() for l in f if not l.startswith(">") and l.strip()]
        return seqs

    print("Downloading human SwissProt proteome ...")
    r = requests.get(UNIPROT_HUMAN_URL, stream=True, timeout=600)
    r.raise_for_status()
    seqs, current = [], []
    lines_written = 0
    with open(cache_path, "w") as out:
        for line in r.iter_lines(decode_unicode=True):
            out.write(line + "\n")
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                current = []
            else:
                current.append(line.strip())
        if current:
            seqs.append("".join(current))
    print(f"Downloaded {len(seqs)} human proteins.")
    return seqs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache) if args.cache else out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load IEDB T-cell data
    # ------------------------------------------------------------------
    df_raw = load_iedb_tcell(cache / "tcell_full_v3.csv", args.iedb_file)
    df = flatten_columns(df_raw.copy())

    print(f"IEDB raw rows: {len(df):,}")

    # IEDB v3 column names after flatten_columns() (spaces → underscores, joined by |)
    # These are fixed for tcell_full_v3; keep a fallback fuzzy search for resilience.
    KNOWN_COLS = {
        "epitope":    "Epitope|Name",
        "host":       "Host|Name",
        "allele":     "MHC_Restriction|Name",
        "mhc_class":  "MHC_Restriction|Class",
        "qualit":     "Assay|Qualitative_Measurement",
        "assay_type": "Assay|Method",
        "obj_type":   "Epitope|Object_Type",
    }

    def find_col(known: str, fallback_keywords: list[str]) -> str | None:
        if known in df.columns:
            return known
        for kw in fallback_keywords:
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    col_epitope    = find_col(KNOWN_COLS["epitope"],   ["Epitope|Name", "Linear_Sequence", "Description"])
    col_host       = find_col(KNOWN_COLS["host"],      ["Host|Name", "Host_Organism"])
    col_allele     = find_col(KNOWN_COLS["allele"],    ["MHC_Restriction|Name", "Allele_Name", "MHC_Allele"])
    col_mhc_class  = find_col(KNOWN_COLS["mhc_class"], ["MHC_Restriction|Class", "MHC_Class", "allele_class"])
    col_qualit     = find_col(KNOWN_COLS["qualit"],    ["Qualitative_Measurement", "Qualitative_Measure"])
    col_assay_type = find_col(KNOWN_COLS["assay_type"],["Assay|Method", "Method/Technique", "Assay_Type"])
    col_object_type= find_col(KNOWN_COLS["obj_type"],  ["Object_Type"])

    for name, col in [
        ("epitope", col_epitope), ("host", col_host),
        ("allele", col_allele), ("qualitative", col_qualit),
    ]:
        if col is None:
            sys.exit(f"ERROR: could not find '{name}' column in IEDB CSV.\n"
                     f"Available columns:\n{list(df.columns)}")

    print(f"Column mapping: epitope={col_epitope}, host={col_host}, "
          f"allele={col_allele}, class={col_mhc_class}, qualit={col_qualit}")

    col_response = next(
        (c for c in df.columns if "response_measured" in c.lower()), None
    )
    print(f"Response column: {col_response}")

    # ------------------------------------------------------------------
    # 2. Basic filters shared by immunogenic and Treg sets
    # ------------------------------------------------------------------
    mask_human = df[col_host].str.contains("Homo sapiens", na=False, case=False)
    mask_linear = True  # accept all; filter by valid AA later
    if col_object_type:
        mask_linear = df[col_object_type].str.contains(
            "Linear peptide|Peptide", na=False, case=False
        )

    # MHC-II restriction: use class column if present, else parse allele name
    if col_mhc_class:
        mask_mhc2 = df[col_mhc_class].astype(str).str.contains("II", na=False)
    else:
        df["_mhc2_gene"] = df[col_allele].apply(extract_mhc2_gene)
        mask_mhc2 = df["_mhc2_gene"].notna()

    df_mhc2_human = df[mask_human & mask_linear & mask_mhc2].copy()
    print(f"After MHC-II + human filter: {len(df_mhc2_human):,} rows")

    # ------------------------------------------------------------------
    # 3. Immunogenic set — effector T-cell readouts only
    # ------------------------------------------------------------------
    # Restrict to canonical effector responses (IFN-γ, proliferation, cytotoxicity).
    # Using all positive assays would include IL-10+ entries, which almost entirely
    # overlap with the Treg set (same peptide, different experimental context).
    mask_effector = df_mhc2_human[col_response].isin(EFFECTOR_RESPONSES) if col_response else True
    mask_positive = df_mhc2_human[col_qualit].isin(POSITIVE_QUALITATIVE) & mask_effector
    df_immuno = df_mhc2_human[mask_positive][[col_epitope, col_allele, col_qualit]].copy()
    df_immuno.columns = ["peptide", "allele", "qualitative"]
    df_immuno["peptide"] = df_immuno["peptide"].astype(str).str.strip().str.upper()
    df_immuno = df_immuno[df_immuno["peptide"].apply(is_valid_peptide)]
    df_immuno = df_immuno.drop_duplicates("peptide")
    df_immuno["label"] = "immunogenic"
    print(f"Immunogenic set: {len(df_immuno):,} unique 15-mers")

    # ------------------------------------------------------------------
    # 4. Treg set — tolerance/suppression signal via Assay|Response_measured
    # ------------------------------------------------------------------
    # Use the structured IEDB field rather than keyword-matching free text.
    # IL-10 and TGF-β are canonical immunosuppressive cytokines; suppression/tolerance
    # are explicit IEDB assay outcome categories.
    if col_response:
        mask_treg = (
            df_mhc2_human[col_response].isin(TREG_RESPONSES)
            & df_mhc2_human[col_qualit].isin(POSITIVE_QUALITATIVE)
        )
        extra_cols = [c for c in [col_response] if c in df_mhc2_human.columns]
    else:
        print("WARNING: could not find Response_measured column; Treg set will be empty.")
        mask_treg = pd.Series(False, index=df_mhc2_human.index)
        extra_cols = []

    df_treg = df_mhc2_human[mask_treg][[col_epitope, col_allele, col_qualit] + extra_cols].copy()
    df_treg.columns = ["peptide", "allele", "qualitative"] + (["response"] if extra_cols else [])
    df_treg["peptide"] = df_treg["peptide"].astype(str).str.strip().str.upper()
    df_treg = df_treg[df_treg["peptide"].apply(is_valid_peptide)]
    df_treg = df_treg.drop_duplicates("peptide")
    # Remove sequences that also appear in the immunogenic set
    df_treg = df_treg[~df_treg["peptide"].isin(df_immuno["peptide"])]
    df_treg["label"] = "treg"
    print(f"Treg/tolerogenic set: {len(df_treg):,} unique 15-mers")

    # ------------------------------------------------------------------
    # 5. Human non-immunogenic set
    # ------------------------------------------------------------------
    human_seqs = load_human_proteome(cache / "human_swissprot.fasta")
    immunogenic_set = set(df_immuno["peptide"])

    print("Extracting 15-mers from human proteome ...")
    nonim_peptides: set[str] = set()
    for seq in tqdm(human_seqs):
        for pep in slide_15mers(seq):
            if pep not in immunogenic_set:
                nonim_peptides.add(pep)

    # Subsample to keep dataset balanced (cap at args.nonim_cap)
    nonim_list = list(nonim_peptides)
    if args.nonim_cap and len(nonim_list) > args.nonim_cap:
        rng = np.random.default_rng(42)
        nonim_list = list(rng.choice(nonim_list, size=args.nonim_cap, replace=False))

    df_nonim = pd.DataFrame({"peptide": nonim_list})
    df_nonim["allele"] = ""
    df_nonim["qualitative"] = ""
    df_nonim["label"] = "human_nonim"
    print(f"Human non-immunogenic set: {len(df_nonim):,} 15-mers (after cap)")

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    df_immuno.to_csv(out / "immunogenic.csv", index=False)
    df_treg.to_csv(out / "treg.csv", index=False)
    df_nonim.to_csv(out / "human_nonim.csv", index=False)

    df_combined = pd.concat([df_immuno, df_treg, df_nonim], ignore_index=True)
    df_combined.to_csv(out / "combined.csv", index=False)

    stats = (
        f"IEDB T-cell rows (raw):              {len(df):>10,}\n"
        f"After MHC-II + human filter:         {len(df_mhc2_human):>10,}\n"
        f"Immunogenic 15-mers (effector only): {len(df_immuno):>10,}\n"
        f"Treg/tolerogenic 15-mers:            {len(df_treg):>10,}\n"
        f"  NOTE: Treg set is small (~30). Many IL-10/TGFb peptides also induce\n"
        f"  effector responses in other donors (biological ambiguity). Supplement\n"
        f"  from Tregitope literature (De Groot/EpiVax) before DPO training.\n"
        f"Human non-immunogenic 15-mers:       {len(df_nonim):>10,}\n"
        f"Combined total:                      {len(df_combined):>10,}\n"
    )
    print("\n" + stats)
    (out / "stats.txt").write_text(stats)
    print(f"Output written to {out}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mine IEDB MHC-II immunogenicity data for WP1."
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory to write output CSVs.",
    )
    parser.add_argument(
        "--cache", default=None,
        help="Directory for raw downloaded files (default: <output>/cache).",
    )
    parser.add_argument(
        "--iedb_file", default=None,
        help=(
            "Path to locally downloaded tcell_full_v3.zip (or extracted .csv). "
            "Download from https://www.iedb.org/database_export_v3.php — "
            "IEDB requires accepting terms of service via browser."
        ),
    )
    parser.add_argument(
        "--nonim_cap", type=int, default=500_000,
        help="Max human non-immunogenic 15-mers to include (default: 500000).",
    )
    args = parser.parse_args()
    main(args)
