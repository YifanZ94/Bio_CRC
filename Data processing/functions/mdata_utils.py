import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import scirpy as ir
import anndata as ad
from typing import List, Optional, Sequence
import mudata as md
import scipy.sparse as sp
import copy

import pdb

# Keep intersect cells across mods in one mdata
def inner_cells_per_mdata(m: mu.MuData, mods: Optional[Sequence[str]] = None) -> mu.MuData:
    """Restrict each modality (and m.obs) to cells present in ALL selected modalities."""
    mods = list(mods) if mods is not None else list(m.mod.keys())
    # intersection of obs_names across chosen modalities
    common = None
    for mod in mods:
        if mod not in m.mod:
            raise KeyError(f"Modality '{mod}' not found in MuData (has: {list(m.mod.keys())})")
        idx = m.mod[mod].obs_names
        common = idx if common is None else common.intersection(idx)
    # subset each modality and top-level obs
    m2 = m.copy()
    for mod in list(m2.mod.keys()):
        if mod in mods:
            m2.mod[mod] = m2.mod[mod][common].copy()
        else:
            # Optionally drop non-requested modalities to guarantee consistency
            m2.mod.pop(mod)
    m2.update()  # sync shapes
    return m2


def harmonize_obs_columns(md_list: List[mu.MuData], keep="common") -> List[mu.MuData]:
    """
    Align .obs columns across MuDatas.
    keep='common' -> keep only columns present in ALL MuDatas (recommended here).
    """
    if keep not in {"common", "union"}:
        raise ValueError("keep must be 'common' or 'union'")
    # figure out columns to keep
    cols_sets = [set(m.obs.columns) for m in md_list]
    if keep == "common":
        keep_cols = set.intersection(*cols_sets) if cols_sets else set()
    else:
        keep_cols = set.union(*cols_sets) if cols_sets else set()
    # coerce dtypes consistently (avoid category-mismatch issues)
    out = []
    for m in md_list:
        obs = m.obs.loc[:, sorted(keep_cols)].copy()
        for c in obs.columns:
            # unify categories as strings to avoid category alignment problems
            if pd.api.types.is_categorical_dtype(obs[c]):
                obs[c] = obs[c].astype("string")
            elif pd.api.types.is_object_dtype(obs[c]):
                obs[c] = obs[c].astype("string")
        m2 = m.copy()
        m2._obs = obs  # set without reindexing cells
        out.append(m2)
    return out

def merge_mdatas(
    mdatas: List[mu.MuData], mods: Optional[Sequence[str]] = None,
    keep_obs_columns: str = "common",   # 'common' per user request
    index_unique: Optional[str] = None, # set to '-' to disambiguate duplicate cell IDs
    join_method = "outer",
) -> mu.MuData:

    if len(mdatas) == 0:
        raise ValueError("No MuData objects provided.")

    # 1) ensure inner-cell intersection per MuData
    mdatas_inner = [inner_cells_per_mdata(m, mods=mods) for m in mdatas]

    # 2) harmonize obs columns across mdatas (keep only common columns)
    mdatas_harmonized = harmonize_obs_columns(mdatas_inner, keep=keep_obs_columns)

    # 3) Concatenate by OUTER join on vars (union of genes per modality)
    #    mu.concat forwards to anndata.concat per-modality with join='outer'
    merged = md.concat(
        mdatas_harmonized,
        join= join_method,          # OUTER on vars (genes/features)
        label=None,
        keys=None,
        index_unique=index_unique,  # keep original IDs unless you need disambiguation
        merge="unique",        # safer obs merge for identical columns
    )

    # Optional: sort var names per modality for consistency
    for mod in merged.mod:
        merged.mod[mod].var = merged.mod[mod].var.sort_index()
    merged.update()
    return merged


def sync_mdata_obs(mdata):
# Keep same vars across all mods in one mdata
    # sync obs across mods
    idx1 = mdata.mod["airr"].obs.index
    idx2 = mdata.mod["gex"].obs.index
    common_cells = idx1.intersection(idx2)

    # Subset gex to only include these cells
    gex_subset = mdata.mod["gex"][common_cells, :].copy()
    tcr_subset = mdata.mod["airr"][common_cells, :].copy()
    
    # Create a new MuData object to preserve alignment
    mdata_common = mu.MuData({"gex": gex_subset, "airr": tcr_subset})
    mdata_common.obs = mdata.obs.loc[common_cells]
    
    return mdata_common

# Merge new anndata var to ref anndata
def merge_anndata_to_base(anndata1, anndata2):
    c_gene = anndata2.var_names.intersection(anndata1.var_names)
    print(len(c_gene))
    missing_genes = anndata1.var_names.difference(anndata2.var_names)

    # Pad missing genes with zeros (assumes dense arrays, can be adapted for sparse)
    if len(missing_genes) > 0:
        import pandas as pd
        import scipy.sparse

        shape = (anndata2.n_obs, len(missing_genes))
        X_pad = np.zeros(shape, dtype=anndata2.X.dtype)
        X_pad = scipy.sparse.csr_matrix(X_pad) if isinstance(anndata2.X, scipy.sparse.spmatrix) else X_pad

        # Create dummy .var for missing genes
        var_pad = pd.DataFrame(index=missing_genes)

        # Create a temporary AnnData with padded genes
        adata_pad = ad.AnnData(X=X_pad, obs=anndata2.obs.copy(), var=var_pad)

        # Add missing genes and reorder to match anndata1
        anndata2_full = ad.concat([anndata2, adata_pad], axis=1)
        anndata2_full = anndata2_full[:, anndata1.var_names]
        
        # # Sync var DataFrame: preserve anndata2.var for existing genes, use reference for missing genes
        # # Start with anndata2.var and reindex to match anndata1.var_names
        # var_synced = anndata2.var.reindex(anndata1.var_names)
        # # For missing genes, fill with reference var information
        # missing_genes_var = var_synced.index.difference(anndata2.var.index)
        # if len(missing_genes_var) > 0:
        #     # Add missing columns from reference if needed
        #     for col in anndata1.var.columns:
        #         if col not in var_synced.columns:
        #             var_synced[col] = None
        #     # Fill missing genes with reference var data
        #     var_synced.loc[missing_genes_var] = anndata1.var.loc[missing_genes_var]
        # anndata2_full.var = var_synced
        
        # Explicitly preserve all metadata from anndata2 after concat and reordering
        anndata2_full.obs = anndata2.obs.copy()
        if hasattr(anndata2, 'obsm') and anndata2.obsm is not None:
            anndata2_full.obsm = anndata2.obsm.copy()
        if hasattr(anndata2, 'obsp') and anndata2.obsp is not None:
            anndata2_full.obsp = anndata2.obsp.copy()
        if hasattr(anndata2, 'uns') and anndata2.uns is not None:
            anndata2_full.uns = anndata2.uns.copy()
        if hasattr(anndata2, 'layers') and anndata2.layers is not None:
            anndata2_full.layers = anndata2.layers.copy()

    else:
        anndata2_full = anndata2[:, anndata1.var_names]
        # Sync var DataFrame: preserve anndata2.var (all genes exist, just reordered)
        var_synced = anndata2.var.reindex(anndata1.var_names)
        # Add any missing columns from reference
        for col in anndata1.var.columns:
            if col not in var_synced.columns:
                var_synced[col] = None
                var_synced[col] = anndata1.var[col]
        anndata2_full.var = var_synced
    return anndata2_full

# Merge new mdata to ref mdata, sync genes with the ref mdata, keep all obs etc.
def merge_mdata_to_base(mdata1, mdata2):
    """
    Merge mdata2 to match mdata1's var_names for each modality.
    Pads missing genes with zeros and preserves all metadata from mdata2.
    """
    # Create a new mdata with synced modalities
    synced_mods = {}
    
    # Sync each modality that exists in both mdatas
    for mod_name in mdata1.mod.keys():
        if mod_name in mdata2.mod:
            synced_mods[mod_name] = merge_anndata_to_base(
                mdata1.mod[mod_name], 
                mdata2.mod[mod_name]
            )
        else:
            print(f"Warning: Modality '{mod_name}' not found in mdata2, skipping")
    
    # Create new mdata with synced modalities
    mdata2_synced = mu.MuData(synced_mods)
    
    # Preserve all top-level metadata from mdata2
    mdata2_synced.obs = mdata2.obs.copy()
    if hasattr(mdata2, 'obsm') and mdata2.obsm is not None:
        mdata2_synced.obsm = mdata2.obsm.copy()
    if hasattr(mdata2, 'obsp') and mdata2.obsp is not None:
        mdata2_synced.obsp = mdata2.obsp.copy()
    if hasattr(mdata2, 'uns') and mdata2.uns is not None:
        mdata2_synced.uns = mdata2.uns.copy()
    
    # Update to sync shapes
    mdata2_synced.update()
    return mdata2_synced

def pp_EAE(mdata, celltype_score = 0.4, cellstate_score = 0.4, topN_variable = None):
    prefixes_to_remove = ('CMO', 'ENSM',
                      'Trav', 'Traj', 'Trac', 'Trbv', 'Trbj', 
                     'Trbc', 'Trdv', 'Trdj', 'Trdc', 'Trgv', 'Trgj', 'Trgc') 

    # Filter TCR genes from gex modality only (not top-level mdata)
    TCR_gene_mask = mdata['gex'].var_names.str.startswith(prefixes_to_remove)
    mdata.mod['gex'] = mdata['gex'][:, ~TCR_gene_mask].copy()

    mdata['gex'].var_names_make_unique()
    
    sc.pp.normalize_total(mdata['gex'])
    sc.pp.log1p(mdata['gex'])

    mdata['gex'].var["mt"] = mdata['gex'].var_names.str.startswith("mt-")
    # ribosomal genes
    mdata['gex'].var["ribo"] = mdata['gex'].var_names.str.startswith(("Rps", "Rpl"))
    # hemoglobin genes
    mdata['gex'].var["hb"] = mdata['gex'].var_names.str.contains("^Hb[^(p)]")
    sc.pp.calculate_qc_metrics(
        mdata['gex'], qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )
    
    sc.pp.filter_cells(mdata['gex'], min_genes= 300)
    sc.pp.filter_cells(mdata['gex'], max_genes= 7000)
    sc.pp.filter_cells(mdata['gex'], min_counts=1000)
    sc.pp.filter_cells(mdata['gex'], max_counts=40000)
    sc.pp.filter_genes(mdata['gex'], min_cells=50)

    ## markers 
    type_sets = {
        "CD4"     : ['Cd4'],
        "CD8"     : ['Cd8a', 'Cd8b1', 'Nkg7'],
    }

    CD4_cell_types = {
        "Treg"    : ['Foxp3', 'Il2ra', 'Ctla4', 'Ikzf2'],
        "Th1"     : ['Tbx21', 'Stat1', 'Stat4', 'Ifng', 'Tnf', 'Il2', 'Cxcr3', 'Ccr5', 'Il12rb1', 'Il12rb2', 'Il18r1'],
        "Th2"     : ['Gata3', 'Stat6', 'Il4', 'Il5', 'Il13', 'Ccr4', 'Ccr8', 'Il1rl1', 'Ptgdr2', 'Areg'],
        "Th17"    : ['Rorc', 'Rora', 'Stat3', 'Batf', 'Il17a', 'Il17f', 'Il22', 'Il21', 'Ccr6', 'Il23r'],
    }

    common_states = {
        "Naive"         : ['Ccr7', 'Sell', 'Tcf7', 'Lef1', 'Il7r'],   # both CD4 and CD8 naive
        "Cytotoxic"     : ['Gzmb', 'Gzma', 'Gzmk', 'Prf1', 'Nkg7'],  # Tem_CD8, SLEC, Tex_term, Th1-cytotoxic
        "Exhaustion": ['Pdcd1', 'Tox', 'Lag3', 'Tigit', 'Havcr2'],# shared across Tpex, Tex_int, Tex_term
        "Tcf7_stem"     : ['Tcf7', 'Bcl2', 'Id3', 'Bach2'],           # Tpex, MPEC, Naive
        "Proliferating" : ['Mki67', 'Top2a', 'Tyms', 'Cdk1'],         # any cycling T cell
        "IFN_stim"      : ['Isg15', 'Ifit1', 'Ifit3', 'Mx1', 'Oas1a', 'Gbp2'],
        "Early_activ"   : ['Cd69','Cd28', 'Icos', 'Nr4a1', 'Nr4a2', 'Fos', 'Jun'],
        "Memory_core"   : ['Il7r', 'Bcl2', 'S100a4', 'Ccl5'],         # Tcm, Tem, MPEC
        "Effector" : ['Cx3cr1', 'S1pr1', 'Zeb2', 'Tbx21'],       # SLEC, Tem_CD8, Tex_KLR
    }
    
    type_score_names = []
    subtype_score_names = []
    state_score_names = []
    
    var_names = set(mdata['gex'].var_names)
       
    ### 1) Score and assign main cell types (CD4/CD8)
    for name, val in type_sets.items():
        valid_genes = [g for g in val if g in var_names]
        if valid_genes:
            sc.tl.score_genes(mdata['gex'], gene_list=valid_genes, score_name=name+'_score')
            type_score_names.append(name+'_score')
        else:
            print(f"Warning: No valid genes found for {name} scoring, skipping. Missing genes: {val}")
     
    if type_score_names:
        scores = mdata['gex'].obs[type_score_names]
        best_type = scores.idxmax(axis=1).str.replace(r'_score$', '', regex=True)
        keep = scores.max(axis=1) > celltype_score
        mdata['gex'].obs['cell_type'] = best_type.where(keep, np.nan)
    else:
        print("Warning: No cell type scores could be computed. Setting cell_type to NaN.")
        mdata['gex'].obs['cell_type'] = np.nan
    
    ### 2) For CD4 cells, score subtypes and append to cell_type
    for name, val in CD4_cell_types.items():
        valid_genes = [g for g in val if g in var_names]
        if valid_genes:
            sc.tl.score_genes(mdata['gex'], gene_list=valid_genes, score_name=name+'_score')
            subtype_score_names.append(name+'_score')
        else:
            print(f"Warning: No valid genes found for {name} scoring, skipping. Missing genes: {val}")
    
    cd4_mask = mdata['gex'].obs['cell_type'] == 'CD4'
    if cd4_mask.any() and subtype_score_names:
        subtype_scores = mdata['gex'].obs.loc[cd4_mask, subtype_score_names]
        best_subtype = subtype_scores.idxmax(axis=1).str.replace(r'_score$', '', regex=True)
        subtype_keep = subtype_scores.max(axis=1) > celltype_score
        # Append subtype to CD4 (e.g., "CD4_Treg")
        mdata['gex'].obs.loc[cd4_mask, 'cell_type'] = (
            'CD4_' + best_subtype.where(subtype_keep, '')
        ).str.rstrip('_')
    elif cd4_mask.any() and not subtype_score_names:
        print("Warning: No CD4 subtype scores could be computed. CD4 cells will not have subtype annotation.")
    
    ### 3) Assign cell states using common_states
    for name, val in common_states.items():
        valid_genes = [g for g in val if g in var_names]
        if valid_genes:
            sc.tl.score_genes(mdata['gex'], gene_list=valid_genes, score_name=name+'_score')
            state_score_names.append(name+'_score')
        else:
            print(f"Warning: No valid genes found for {name} scoring, skipping. Missing genes: {val}")
     
    if state_score_names:
        scores = mdata['gex'].obs[state_score_names]
        best_state = scores.idxmax(axis=1).str.replace(r'_score$', '', regex=True)
        keep = scores.max(axis=1) > cellstate_score
        mdata['gex'].obs['state'] = best_state.where(keep, np.nan)
    else:
        print("Warning: No cell state scores could be computed. Setting state to NaN.")
        mdata['gex'].obs['state'] = np.nan   
    
    mdata.obs['state'] = mdata['gex'].obs['state']
    mdata.obs['cell_type'] = mdata['gex'].obs['cell_type']
    mdata.obs['sample_id'] = mdata['gex'].obs['sample_id']
    
    if topN_variable is not None:
        sc.pp.highly_variable_genes(mdata['gex'], n_top_genes=topN_variable)
        mdata.mod['gex'] = mdata['gex'][:, mdata['gex'].var['highly_variable']].copy()

        mdata = sync_mdata_obs(mdata)
    else:
        pass
    
    # pdb.set_trace()
    
    return mdata


def aggregate_mdata_by_clone(
    mdata: mu.MuData,
    clone_col: str = "clone_id",
    gex_mod: str = "gex",
    airr_mod: str = "airr",
) -> mu.MuData:
    """
    Build a clone-level MuData: mean gene expression per clone + one AIRR row per clone.

    - GEX: rows = unique clones, X = mean expression over all cells belonging to that clone
      (denominator = number of cells in the clone). Uses the same ``.var`` as ``mdata[gex_mod]``.
    - AIRR: one observation per clone, taken as the **first** cell in each clone group
      (preserves original ``airr`` ``.obs`` columns and ``.var`` / ``X`` layout for that cell).

    Top-level ``mdata.obsm`` matrices aligned with cell rows are **clone-averaged** (mean over
    cells in each clone). ``mdata.uns`` is **deep-copied** to the output.

    **obs_names**: the original cell barcode of the representative cell (unchanged).
    **``mdata[gex_mod].obs``**: gex-modality columns for the rep cell + ``n_cells_in_clone``
    and ``clone_col``.  No airr columns are mixed in.
    **``mdata[airr_mod].obs``**: airr-modality columns for the rep cell, unchanged.
    **``mdata.obs``** (top-level): all gex columns prefixed ``gex:`` joined with all airr
    columns prefixed ``airr:``.

    ``mdata.obsp`` is not transferred (cell--cell graphs are not defined at clone level here).

    Requires aligned cell indices between ``gex`` and ``airr``. Call ``sync_mdata_obs`` first if needed.
    """
    m = inner_cells_per_mdata(mdata, mods=[gex_mod, airr_mod])
    if clone_col not in m[airr_mod].obs.columns:
        raise KeyError(f"{clone_col!r} not found in m['{airr_mod}'].obs")

    ad_g = m[gex_mod]
    ad_a = m[airr_mod]
    clones = ad_a.obs[clone_col].astype(str)
    bad = clones.isin(["", "nan", "None", "nan", "NaN"])
    if bad.any():
        clones = clones.mask(bad, np.nan)

    # Drop cells with missing clone label for aggregation
    keep = clones.notna()
    if not keep.all():
        m = m[keep.values].copy()
        ad_g = m[gex_mod]
        ad_a = m[airr_mod]
        clones = ad_a.obs[clone_col].astype(str)

    obs_idx = ad_g.obs_names
    grp = pd.Series(clones.values, index=obs_idx, name="_clone_agg")
    uniq = pd.Index(sorted(pd.unique(grp.dropna().values)))

    counts = grp.value_counts().reindex(uniq).fillna(0).astype(int)
    n_cells_per_clone = counts.values

    rep_map = grp.groupby(grp, sort=False).apply(lambda s: s.index[0]).to_dict()
    rep_index = [rep_map[c] for c in uniq]

    # Map rep cell barcode → clone-level values (index stays as cell barcodes)
    rep_to_count = {rep_map[c]: int(counts.loc[c]) for c in uniq}
    rep_to_clone = {rep_map[c]: c for c in uniq}

    # ── GEX obs: only gex columns, obs_names = rep cell barcodes ─────────────
    gex_obs = ad_g.obs.loc[rep_index].copy()
    gex_obs["n_cells_in_clone"] = [rep_to_count[r] for r in rep_index]
    gex_obs[clone_col] = [rep_to_clone[r] for r in rep_index]

    # ── AIRR obs: only airr columns, obs_names = rep cell barcodes ────────────
    airr_obs = ad_a.obs.loc[rep_index].copy()

    # ── Top-level obs: gex: + airr: prefixed columns ──────────────────────────
    obs_join = gex_obs.rename(columns=lambda c: f"gex:{c}").join(
        airr_obs.rename(columns=lambda c: f"airr:{c}"), how="left"
    )

    X = ad_g.X
    mean_rows = []
    for c in uniq:
        mask = (grp == c).values
        if not np.any(mask):
            continue
        sub = X[mask, :]
        if sp.issparse(sub):
            mean_rows.append(np.asarray(sub.mean(axis=0)).ravel())
        else:
            mean_rows.append(np.asarray(sub.mean(axis=0)).ravel())
    X_clone = np.vstack(mean_rows)

    ad_g_clone = ad.AnnData(
        X=X_clone,
        obs=gex_obs.copy(),
        var=ad_g.var.copy(),
    )
    ad_g_clone.uns = copy.deepcopy(dict(ad_g.uns))

    # airr: rep cells, obs = airr-only columns, obs_names = rep cell barcodes
    ad_a_clone = ad_a[rep_index].copy()
    ad_a_clone.obs = airr_obs.copy()
    ad_a_clone.uns = copy.deepcopy(dict(ad_a.uns))

    out = mu.MuData({gex_mod: ad_g_clone, airr_mod: ad_a_clone})
    out.obs = obs_join.copy()

    out.uns = copy.deepcopy(dict(m.uns))
    for k, mat in dict(m.obsm).items():
        try:
            arr = mat.toarray() if sp.issparse(mat) else np.asarray(mat)
            if arr.shape[0] != len(grp):
                raise ValueError(f"obsm row count {arr.shape[0]} != n_obs {len(grp)}")
            rows = []
            for c in uniq:
                mask = (grp == c).values
                if not np.any(mask):
                    if arr.ndim > 1:
                        rows.append(np.full(arr.shape[1:], np.nan))
                    else:
                        rows.append(np.nan)
                    continue
                blk = arr[mask]
                if arr.ndim == 1:
                    rows.append(np.nanmean(blk))
                else:
                    rows.append(np.nanmean(blk, axis=0))
            if arr.ndim == 1:
                out.obsm[k] = np.asarray(rows).reshape(-1, 1)
            else:
                out.obsm[k] = np.vstack(rows)
        except Exception as exc:
            print(f"[mdata_utils] skipped obsm[{k!r}] at clone level: {exc}")

    out.update()
    return out


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Univariate OLS slope of ``y`` on ``x`` (both 1-D). Returns NaN if undefined."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    xm = x - np.nanmean(x)
    ssx = np.nansum(xm * xm)
    if ssx <= 0:
        return float("nan")
    ym = y - np.nanmean(y)
    return float(np.nansum(xm * ym) / ssx)


def per_clone_gene_slope_mdata(
    mdata: mu.MuData,
    clone_col: str = "clone_id",
    gex_mod: str = "gex",
    airr_mod: str = "airr",
    min_nonzero_cells: int = 2,
    min_cells_per_clone: int = 3,
    slope_x_col: Optional[str] = None,
    layer: Optional[str] = None,
) -> mu.MuData:
    """
    For each clone, keep genes that are **strictly** nonzero in **more than** ``min_nonzero_cells``
    cells (within that clone), then fit **y ~ x** where ``y`` is the gene's expression in each cell
    and ``x`` is either:

    - ``m.obs[slope_x_col]`` (must be numeric, same cells as ``gex``), or
    - if ``slope_x_col`` is ``None``, ``x = 0, 1, …, n-1`` in the order cells appear in
      ``adata.obs_names`` for that clone (a positional axis — supply ``slope_x_col`` for a
      meaningful covariate such as pseudotime or library size).

    Slopes are stored in ``out[gex_mod].X`` with shape **(n_clones, n_genes)**. Entries are **0**
    when the gene did not pass the per-clone nonzero filter (``nnz <= min_nonzero_cells``) or when
    the slope is undefined (e.g. constant ``x``). The matrix is always NaN-free.

    **Metadata layout:**

    - **obs_names**: original rep-cell barcode (unchanged throughout).
    - ``out[gex_mod].obs``: gex-modality columns for the rep cell + ``n_cells_in_clone``,
      ``clone_col``, ``slope_skipped``.  No airr columns mixed in.
    - ``out[airr_mod].obs``: airr-modality columns for the rep cell, unchanged.
    - ``out.obs`` (top-level): gex columns prefixed ``gex:`` joined with airr columns
      prefixed ``airr:``.
    - ``out.uns``: deep copy of ``m.uns`` with ``per_clone_gene_slope`` params added.
    - ``out.obsm``: representative cell row per clone from each ``m.obsm`` key (no averaging).
    """
    m = inner_cells_per_mdata(mdata, mods=[gex_mod, airr_mod])
    if clone_col not in m[airr_mod].obs.columns:
        raise KeyError(f"{clone_col!r} not found in m['{airr_mod}'].obs")
    if slope_x_col is not None and slope_x_col not in m.obs.columns:
        raise KeyError(f"{slope_x_col!r} not found in m.obs")

    # Use ``adata_g`` (not ``ad``) so we do not shadow ``import anndata as ad``.
    adata_g = m[gex_mod]
    adata_a = m[airr_mod]
    clones = adata_a.obs[clone_col].astype(str)
    bad = clones.isin(["", "nan", "None", "nan", "NaN"])
    if bad.any():
        clones = clones.mask(bad, np.nan)
    keep = clones.notna()
    if not keep.all():
        m = m[keep.values].copy()
        adata_g = m[gex_mod]
        adata_a = m[airr_mod]
        clones = adata_a.obs[clone_col].astype(str)

    obs_names = adata_g.obs_names
    grp = pd.Series(clones.values, index=obs_names, name="_clone_slope")
    uniq = pd.Index(sorted(pd.unique(grp.dropna().values)))

    counts = grp.value_counts().reindex(uniq).fillna(0).astype(int)

    # Representative cell per clone (first cell in sorted obs order).
    rep_map = grp.groupby(grp, sort=False).apply(lambda s: s.index[0]).to_dict()
    rep_index = [rep_map[c] for c in uniq]

    # Map rep cell barcode → clone-level values (index stays as cell barcodes)
    rep_to_count = {rep_map[c]: int(counts.loc[c]) for c in uniq}
    rep_to_clone = {rep_map[c]: c for c in uniq}

    # ── GEX obs: only gex columns, obs_names = rep cell barcodes ─────────────
    obs_df = adata_g.obs.loc[rep_index].copy()
    obs_df["n_cells_in_clone"] = [rep_to_count[r] for r in rep_index]
    obs_df[clone_col] = [rep_to_clone[r] for r in rep_index]
    obs_df["slope_skipped"] = [rep_to_count[r] < int(min_cells_per_clone) for r in rep_index]

    n_genes = adata_g.n_vars
    # Initialise to 0: genes that don't pass the per-clone nonzero filter
    # or whose slope is undefined are treated as "no slope signal" (0),
    # so the matrix is NaN-free and safe for downstream tools (PCA, etc.).
    slope_mat = np.zeros((len(uniq), n_genes), dtype=np.float64)

    for ri, c in enumerate(uniq):
        n_c = int(counts.loc[c])
        mask = (grp == c).values
        if n_c < min_cells_per_clone:
            continue

        sub = adata_g[mask]
        Xs = sub.layers[layer] if layer is not None else sub.X
        if sp.issparse(Xs):
            Xd = Xs.toarray()
        else:
            Xd = np.asarray(Xs, dtype=np.float64)

        if slope_x_col is None:
            x = np.arange(n_c, dtype=np.float64)
        else:
            idx_cells = sub.obs_names
            x = np.asarray(m.obs.loc[idx_cells, slope_x_col], dtype=np.float64).ravel()

        nnz = np.sum(Xd != 0.0, axis=0)
        use_gene = nnz > int(min_nonzero_cells)

        for gj in np.flatnonzero(use_gene):
            y = Xd[:, gj]
            s = _ols_slope(x, y)
            # _ols_slope returns nan when slope is undefined (e.g. constant x);
            # keep 0 in those cases so X stays NaN-free.
            if not np.isnan(s):
                slope_mat[ri, gj] = s

    ad_out = ad.AnnData(
        X=slope_mat,
        obs=obs_df.copy(),
        var=adata_g.var.copy(),
    )
    ad_out.uns = copy.deepcopy(dict(adata_g.uns))

    # ── AIRR obs: only airr columns, obs_names = rep cell barcodes ────────────
    airr_obs = adata_a.obs.loc[rep_index].copy()
    ad_a_clone = adata_a[rep_index].copy()
    ad_a_clone.obs = airr_obs.copy()
    ad_a_clone.uns = copy.deepcopy(dict(adata_a.uns))

    # ── Top-level obs: gex: + airr: prefixed columns ──────────────────────────
    top_obs = obs_df.rename(columns=lambda c: f"gex:{c}").join(
        airr_obs.rename(columns=lambda c: f"airr:{c}"), how="left"
    )

    out = mu.MuData({gex_mod: ad_out, airr_mod: ad_a_clone})
    out.obs = top_obs.copy()

    # uns: deep copy of m.uns, then add slope parameters.
    out.uns = copy.deepcopy(dict(m.uns))
    out.uns["per_clone_gene_slope"] = {
        "min_nonzero_cells": int(min_nonzero_cells),
        "min_cells_per_clone": int(min_cells_per_clone),
        "slope_x_col": slope_x_col,
        "layer": layer,
    }

    # obsm: representative cell row per clone from each m.obsm key.
    for k, mat in dict(m.obsm).items():
        try:
            arr = mat.toarray() if sp.issparse(mat) else np.asarray(mat)
            out.obsm[k] = arr[
                [obs_names.get_loc(r) for r in rep_index], :
            ] if arr.ndim > 1 else arr[[obs_names.get_loc(r) for r in rep_index]]
        except Exception as exc:
            print(f"[mdata_utils] skipped obsm[{k!r}]: {exc}")

    out.update()
    return out


def select_onecell_per_clone(mdata):
    airr_obs = mdata['airr'].obs.copy()
    clone_id_col = airr_obs['clone_id']
    if hasattr(clone_id_col, 'cat'):
        clone_id_col = clone_id_col.astype(str)
    airr_obs['_clone_id_str'] = clone_id_col

    expanded = airr_obs[airr_obs['clone_id_size'] > 1].dropna(subset=['_clone_id_str'])
    sampled_expanded_idx = (
        expanded
        .groupby('_clone_id_str', observed=True)
        .sample(n=1, random_state=42)
        .index
    )

    single_idx = airr_obs[airr_obs['clone_id_size'] == 1].index
    keep_idx = sampled_expanded_idx.append(single_idx)

    mdata = mdata[keep_idx].copy()
    return mdata