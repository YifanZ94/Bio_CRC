"""
Convert wide-format TCR chain TSV (one row per cell, _TRA/_TRA2/_TRB/_TRB2 suffixes)
to scirpy-compatible filtered_contig_annotations CSV (one row per contig).

Usage:
    python convert_tcr_to_scirpy.py input.tsv output_filtered_contig_annotations.csv

Then load with:
    import scirpy as ir
    adata = ir.io.read_10x_vdj("output_filtered_contig_annotations.csv")
"""

import pandas as pd
import numpy as np
import sys
import os

# ── config ────────────────────────────────────────────────────────────────────
INPUT_TSV  = sys.argv[1] if len(sys.argv) > 1 else "tcr_chain_information.tsv"
OUTPUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "filtered_contig_annotations.csv"

# Each entry: (column_suffix_in_input, canonical_chain_name_for_scirpy)
CHAIN_SLOTS = [
    ("TRA",  "TRA"),
    ("TRA2", "TRA"),
    ("TRB",  "TRB"),
    ("TRB2", "TRB"),
]

# Columns scirpy expects in filtered_contig_annotations
OUTPUT_COLS = [
    "barcode", "is_cell", "contig_id", "high_confidence",
    "length", "chain", "v_gene", "d_gene", "j_gene", "c_gene",
    "full_length", "productive", "cdr3", "cdr3_nt",
    "reads", "umis", "raw_clonotype_id", "raw_consensus_id",
]

# ── helpers ───────────────────────────────────────────────────────────────────
def strip_locus_prefix(value: object, chain: str) -> object:
    """Remove leading 'TRA_' / 'TRB_' prefix from CDR3 AA sequences."""
    if pd.notna(value) and isinstance(value, str):
        return value.replace(f"{chain}_", "")
    return value


def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cell in df.iterrows():
        barcode = cell["barcode_id"]
        for suffix, chain_name in CHAIN_SLOTS:
            contig_col = f"contig_id_{suffix}"
            # skip slots with no contig data
            if contig_col not in df.columns or pd.isna(cell.get(contig_col)):
                continue

            def g(field: str):
                return cell.get(f"{field}_{suffix}", np.nan)

            rows.append({
                "barcode":          barcode,
                "is_cell":          g("is_cell"),
                "contig_id":        g("contig_id"),
                "high_confidence":  g("high_confidence"),
                "length":           g("length"),
                "chain":            chain_name,
                "v_gene":           g("v_gene"),
                "d_gene":           g("d_gene"),
                "j_gene":           g("j_gene"),
                "c_gene":           g("c_gene"),
                "full_length":      g("full_length"),
                "productive":       g("productive"),
                "cdr3":             strip_locus_prefix(g("cdr3"), chain_name),
                "cdr3_nt":          g("cdr3_nt"),
                "reads":            g("reads"),
                "umis":             g("umis"),
                "raw_clonotype_id": g("raw_clonotype_id"),
                "raw_consensus_id": g("raw_consensus_id"),
            })
            

    return pd.DataFrame(rows, columns=OUTPUT_COLS)

