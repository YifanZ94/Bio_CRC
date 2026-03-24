"""
Utilities extracted from `E4S_annotation.ipynb`.

Main capabilities:
1) Run `sc.tl.rank_genes_groups` for Leiden clusters and export top marker genes to CSV.
2) Plot UMAPs for gene sets provided as a dict: {label: [gene1, gene2, ...]}.
3) Score Leiden clusters using marker-gene dicts and assign the best-matching label per cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


def _get_rank_genes_groups_names(result: Mapping[str, Any]) -> Any:
    if "names" not in result:
        raise KeyError("Expected `adata.uns['rank_genes_groups']['names']`.")
    return result["names"]


def _get_rank_genes_groups_groups(result: Mapping[str, Any]) -> List[str]:
    """
    Extract group names from Scanpy's `rank_genes_groups` result object.
    Works for structured arrays (dtype.names) and for dict-like results (keys()).
    """
    names = _get_rank_genes_groups_names(result)

    dtype = getattr(names, "dtype", None)
    dtype_names = getattr(dtype, "names", None)
    if dtype_names:
        return list(dtype_names)

    keys = getattr(names, "keys", None)
    if callable(keys):
        return list(names.keys())

    # Fallback: try index-based names
    # (Last resort; typically scanpy returns structured arrays/dicts.)
    raise TypeError(
        "Unsupported structure for `rank_genes_groups['names']`. "
        "Expected structured array with dtype.names or dict-like with keys()."
    )


def rank_leiden_markers_to_csv(
    adata: Any,
    leiden_col: str = "leiden",
    method: str = "wilcoxon",
    reference: str = "rest",
    groups: Optional[Sequence[str]] = None,
    n_genes: int = 20,
    output_top_genes_csv: Optional[str | Path] = "E4_top_marker_genes_per_cluster.csv",
    output_detailed_csv: Optional[str | Path] = None,
    rank_uns_key: str = "rank_genes_groups",
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run `sc.tl.rank_genes_groups` on `adata` grouped by Leiden and export marker genes.

    Returns:
        (df_top_genes, df_detailed)

    `df_top_genes` is wide: columns are clusters, rows are top genes.
    `df_detailed` is long: one row per (cluster, gene) with scores/pvals/logFC.
    """
    if leiden_col not in adata.obs:
        raise KeyError(f"`{leiden_col}` not found in `adata.obs`.")

    # Scanpy groups default to "all" when `groups=None`. The notebook ran:
    # `sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')`
    if groups is None:
        sc.tl.rank_genes_groups(adata, groupby=leiden_col, method=method)
    else:
        sc.tl.rank_genes_groups(
            adata,
            groupby=leiden_col,
            groups=list(groups),
            reference=reference,
            method=method,
        )

    result = adata.uns[rank_uns_key]
    all_groups = _get_rank_genes_groups_groups(result)
    if groups is not None:
        wanted = set(map(str, groups))
        groups = [g for g in all_groups if str(g) in wanted]
    else:
        groups = all_groups

    # Wide top genes
    df_top_genes = pd.DataFrame({g: result["names"][g][:n_genes] for g in groups})

    # Detailed (long) genes
    required_keys = ["scores", "logfoldchanges", "pvals", "pvals_adj"]
    for k in required_keys:
        if k not in result:
            raise KeyError(f"Expected `{rank_uns_key}['{k}']` to exist.")

    df_detailed = pd.DataFrame()
    for g in groups:
        temp_df = pd.DataFrame(
            {
                "cluster": g,
                "gene": result["names"][g][:n_genes],
                "scores": result["scores"][g][:n_genes],
                "logfoldchanges": result["logfoldchanges"][g][:n_genes],
                "pvals": result["pvals"][g][:n_genes],
                "pvals_adj": result["pvals_adj"][g][:n_genes],
            }
        )
        df_detailed = pd.concat([df_detailed, temp_df], ignore_index=True)

    if output_top_genes_csv is not None:
        out_path = Path(output_top_genes_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_top_genes.to_csv(out_path, index=True)

    if output_detailed_csv is not None:
        out_path = Path(output_detailed_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_detailed.to_csv(out_path, index=False)

    if show_progress:
        print(f"Saved rank_genes_groups top genes for {len(groups)} clusters.")

    return df_top_genes, df_detailed


def plot_umap_for_gene_dict(
    adata: Any,
    gene_dict: Mapping[str, Sequence[str]],
    umap_params: Optional[Mapping[str, Any]] = None,
    *,
    save: bool = False,
    show: bool = True,
) -> Dict[str, List[str]]:
    """
    Plot UMAPs for gene sets from a dict.

    For each entry `{label: [gene1, gene2, ...]}`:
      - filters to genes present in `adata.var_names`
      - calls `sc.pl.umap(adata, color=valid_genes, title=[f\"{label}: {gene}\" ...])`

    Returns:
        dict mapping label -> list of genes actually used (present in var_names).
    """
    if umap_params is None:
        umap_params = {
            "legend_loc": "on data",
            "s": 30,
            "add_outline": False,
            "ncols": 3,
            "frameon": False,
        }

    out: Dict[str, List[str]] = {}
    for label, genes in gene_dict.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if not valid_genes:
            print(f"[Warning] No valid markers for {label}")
            continue

        print(f"UMAP: {label} (markers: {valid_genes})")
        out[label] = valid_genes

        sc.pl.umap(
            adata,
            color=valid_genes,
            title=[f"{label}: {gene}" for gene in valid_genes],
            save=save,
            show=show,
            **dict(umap_params),
        )

    return out

from scipy.stats import rankdata
def score_markers(adata, marker_dict, score_suffix='_score'):
    score_cols_added = []
    n = adata.n_obs

    for marker_name, genes in marker_dict.items():
        # Split into up and down gene lists
        up_genes   = [g for g in genes if not g.endswith('-')]
        down_genes = [g.rstrip('-') for g in genes if g.endswith('-')]

        # Validate against adata.var_names
        valid_up   = [g for g in up_genes   if g in adata.var_names]
        valid_down = [g for g in down_genes if g in adata.var_names]

        if not valid_up and not valid_down:
            print(f"[Warning] No valid genes found for '{marker_name}'. Skipping.")
            continue

        # Warn about missing genes without skipping
        missing = [g for g in up_genes   if g not in adata.var_names] + \
                  [g + '-' for g in down_genes if g not in adata.var_names]
        if missing:
            print(f"[Warning] '{marker_name}': genes not found in adata: {missing}")

        # Rank-based composite score
        scores = np.zeros(n)
        total  = len(valid_up) + len(valid_down)

        for gene in valid_up:
            expr = adata[:, gene].X
            expr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()
            scores += rankdata(expr) / n

        for gene in valid_down:
            expr = adata[:, gene].X
            expr = expr.toarray().flatten() if hasattr(expr, 'toarray') else np.array(expr).flatten()
            scores += (1 - rankdata(expr) / n)

        scores /= total

        score_name = f"{marker_name}{score_suffix}"
        adata.obs[score_name] = scores
        score_cols_added.append(score_name)
        print(f"[Info] Scored '{marker_name}': {len(valid_up)} up, {len(valid_down)} down genes.")

    return score_cols_added

def score_leiden_clusters_by_marker_dict(
    adata: Any,
    marker_dict: Mapping[str, Sequence[str]],
    *,
    leiden_col: str = "leiden",
    score_suffix: str = "_score",
    cell_type_annotation_col: str = "cell_type_annotation",
    add_cell_type_col: bool = True,
    return_cluster_scores: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Score cells using marker genes (with up/down regulation support via '-' suffix),
    then annotate each Leiden cluster with the marker set that has the highest mean score.

    Marker dict convention:
      - plain gene name  -> upregulated (e.g. 'Tcf7')
      - gene name + '-' -> downregulated (e.g. 'Tox-')

    Scoring uses rank-based composite scoring (score_markers), which accounts for
    both upregulated and downregulated markers.
    """
    if leiden_col not in adata.obs:
        raise KeyError(f"`{leiden_col}` not found in `adata.obs`.")

    # Replace sc.tl.score_genes loop with rank-based scorer that handles up/down genes
    score_cols_added = score_markers(adata, marker_dict, score_suffix=score_suffix)

    if not score_cols_added:
        raise ValueError("No marker gene sets produced valid genes; nothing to score.")

    # Compute mean score per Leiden cluster
    cluster_scores = adata.obs.groupby(leiden_col)[score_cols_added].mean()

    # Find the marker set with the highest mean score for each cluster
    cluster_annotations = (
        cluster_scores
        .idxmax(axis=1)
        .str.replace(score_suffix, "", regex=False)
    )
    cluster_annotations.name = cell_type_annotation_col

    if add_cell_type_col:
        adata.obs[cell_type_annotation_col] = adata.obs[leiden_col].map(cluster_annotations)

    if return_cluster_scores:
        return cluster_scores, cluster_annotations

    return pd.DataFrame(), cluster_annotations

def assign_annotation_by_leiden(adata, annotation_dict, annotation_name: str, default_obs: str = None):
    if default_obs is not None and default_obs in adata.obs.columns:
        adata.obs[annotation_name] = adata.obs.apply(
            lambda row: annotation_dict.get(int(row['leiden']), row[default_obs]),
            axis=1
        ).astype('category')
    else:
        adata.obs[annotation_name] = adata.obs.apply(
            lambda row: annotation_dict.get(int(row['leiden']), str('nan')),
            axis=1
        ).astype('category')

    # return adata

def score_cells_by_marker_dict(
    adata: Any,
    marker_dict: Mapping[str, Sequence[str]],
    *,
    score_suffix: str = "_score",
    cell_type_annotation_col: str = "cell_type_annotation",
    add_cell_type_col: bool = True,
    return_cell_scores: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Score each cell using marker genes (with up/down regulation support via '-' suffix)
    and annotate by the highest score per cell.

    Unlike `score_leiden_clusters_by_marker_dict`, this is *not* restricted to
    Leiden clusters. The best label is chosen per cell from the marker-set
    score columns.

    Marker dict convention:
      - plain gene name  -> upregulated (e.g. 'Tcf7')
      - gene name + '-' -> downregulated (e.g. 'Tox-')
    """
    score_cols_added = score_markers(adata, marker_dict, score_suffix=score_suffix)

    if not score_cols_added:
        raise ValueError("No marker gene sets produced valid genes; nothing to score.")

    cell_scores = adata.obs[score_cols_added].copy()
    best_labels = cell_scores.idxmax(axis=1).str.replace(score_suffix, "", regex=False)
    best_labels.name = cell_type_annotation_col

    if add_cell_type_col:
        adata.obs[cell_type_annotation_col] = best_labels

    if return_cell_scores:
        return cell_scores, best_labels

    return pd.DataFrame(), best_labels


        

__all__ = [
    "rank_leiden_markers_to_csv",
    "plot_umap_for_gene_dict",
    "score_leiden_clusters_by_marker_dict",
    "assign_annotation_by_leiden",
    "score_cells_by_marker_dict",
]

