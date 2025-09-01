import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import scirpy as ir

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