import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import scirpy as ir
import anndata as ad
from typing import List, Optional, Sequence
import mudata as md

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
        join="outer",          # OUTER on vars (genes/features)
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

# Keep same vars across all mods in one mdata
def sync_mdata_obs(mdata):
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

    else:
        anndata2_full = anndata2[:, anndata1.var_names]
    return anndata2_full

def pp_EAE(mdata, celltype_score = 0.4, cellstate_score = 0.4, topN_variable = None):
    prefixes_to_remove = ('CMO', 'ENSM',
                      'Trav', 'Traj', 'Trac', 'Trbv', 'Trbj', 
                     'Trbc', 'Trdv', 'Trdj', 'Trdc', 'Trgv', 'Trgj', 'Trgc') 

    TCR_gene_mask = mdata.var_names.str.startswith(prefixes_to_remove)
    mdata = mdata[:, ~TCR_gene_mask]

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
    type_sets = {"CD4" : ['Cd4', 'Cd4a'],
                 "CD8" : ['Cd8a','Cd8b1','Nkg7'],
                "Treg": ['Foxp3', 'Ikzf2', 'Ctla4', 'Il2ra'],
                "Th17" : ['Ccr6', 'Il22', 'Il17', 'Il17a'],
                    }

    state_sets = {"IFN_stim": ['Isg15', 'Gbp2', 'Ifih1'],
                "Activation": ['Icos', 'Cd69', 'Cd28'],
                "Exhaust": ['Pdcd1', 'Lag3', 'Havcr2'],
                "Mem_Naive": ['Ccr7', 'Sell', 'Cd27'],
                 }
    
    type_score_names = []
    state_score_names = []
       
    ### assign cell types
    for name, val in type_sets.items():
        sc.tl.score_genes(mdata['gex'], gene_list=val, score_name= name+'score')
        type_score_names.append(name+'score')
     
    if 'cell_type' not in mdata['gex'].obs:
        mdata['gex'].obs['cell_type'] = np.nan

    # 3) pick the highest-scoring type per cell
    scores = mdata['gex'].obs[type_score_names]
    best_type = scores.idxmax(axis=1).str.replace(r'score$', '', regex=True)

    # apply a global threshold
    keep = scores.max(axis=1) > celltype_score
    best_type = best_type.where(keep)

    # 4) write labels
    mdata['gex'].obs['cell_type'] = mdata['gex'].obs['cell_type'].fillna(best_type)
    
    ### assign cell states
    for name, val in state_sets.items():
        sc.tl.score_genes(mdata['gex'], gene_list=val, score_name= name+'score')
        state_score_names.append(name+'score')
     
    if 'state' not in mdata['gex'].obs:
        mdata['gex'].obs['state'] = np.nan

    scores = mdata['gex'].obs[state_score_names]
    best_type = scores.idxmax(axis=1).str.replace(r'score$', '', regex=True)

    # apply a global threshold
    keep = scores.max(axis=1) > cellstate_score
    best_type = best_type.where(keep)

    # 4) write labels
    mdata['gex'].obs['state'] = mdata['gex'].obs['state'].fillna(best_type)   
    
    mdata.obs['state'] = mdata['gex'].obs['state']
    mdata.obs['cell_type'] = mdata['gex'].obs['cell_type']
    mdata.obs['sample_id'] = mdata['gex'].obs['sample_id']
    
    if topN_variable is not None:
        sc.pp.highly_variable_genes(mdata['gex'], n_top_genes=topN_variable)
        mdata.mod['gex'] = mdata['gex'][:, mdata['gex'].var['highly_variable']].copy()

        mdata = mdata_utils.sync_mdata_obs(mdata)
    else:
        pass
    
    # pdb.set_trace()
    
    return mdata

# def pp_tissue_featured_genes(mdata):
#     mdata = 