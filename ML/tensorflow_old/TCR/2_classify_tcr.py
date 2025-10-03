# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 14:56:18 2025

@author: a4945
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:03:00 2025

# this script predict the 

@author: a4945
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Masking
from tensorflow.keras.optimizers import SGD
from tensorflow.random import set_seed

set_seed(455)

import numpy as np
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
np.random.seed(455)

import pandas as pd

import sys
sys.path.append("/LEE EAE/TCR")

#%%  load data
from sklearn.preprocessing import LabelEncoder

matrix = pd.read_csv('cdr3_id--RotationEncodingBL62.txt_EncodingMatrix.txt', sep = '\t', header = None)
matrix.rename(columns={0:'cdr3_b_aa', 1: 'BC'}, inplace=True)

tcr_rep = pd.read_csv("tcr_rep.csv", delimiter=",")
tcr_rep = tcr_rep.dropna()
    
# num_classes = labels_all.iloc[:,target_idx].value_counts().shape[0]
tcr_rep['L_cdr3_alpha'] = tcr_rep['VJ_1_cdr3_aa'].str.len()

tcr_rep['L_cdr3_beta'] = tcr_rep['VDJ_1_cdr3_aa'].str.len()

#%%
import matplotlib.pyplot as plt

# Step 1: Get unique cell types
cell_types = tcr_rep["tissue"].unique()

spl_num, cns_num = tcr_rep["tissue"].value_counts()

# Step 2: Count class distribution per cell type
 
grouped = tcr_rep.groupby(["tissue", "VDJ_1_v_call"]).size().unstack(fill_value=0)

grouped.iloc[1,:] = (grouped.iloc[1,:]/(spl_num/cns_num)).astype(int)

grouped.loc[2,:] = grouped.iloc[1,:]/grouped.iloc[0,:]

grouped = grouped.loc[:, grouped.loc[2].sort_values(ascending=True).index]
#%%
# Select the relevant categorical columns
# "VJ_1_v_call","VJ_1_j_call", "VDJ_1_v_call", "VDJ_1_j_call", 
#             'L_cdr3_alpha', 'L_cdr3_beta'
            
cat_cols = ["VDJ_1_v_call", "L_cdr3_beta"]

# Get number of unique combinations
num_combinations = tcr_rep[cat_cols].drop_duplicates().shape[0]

print("Number of unique combinations:", num_combinations)


#%%
from sklearn.preprocessing import LabelEncoder

cat_ori = pd.read_csv("obs_classes.csv", delimiter=",")
cat_ori = cat_ori.fillna('sNaN')

cats = pd.merge(cat_ori.iloc[:,:7], tcr_rep, how = 'inner',
                   left_on='Unnamed: 0', right_on='cell_id', )

# for col in merged.columns:
#     merged[col] = merged[col].astype('category').cat.codes

target_idx = 3


#%%
GIANA_emb = pd.read_csv("cdr3_id--RotationEncodingBL62.txt_EncodingMatrix.txt", sep = '\t', header = None)
GIANA_emb.rename(columns={0: 'cdr3_b_aa', 1: 'Unnamed: 0'}, inplace=True)

# concate with the GIANA embs
GIANA_emb_merged = pd.merge(GIANA_emb, cats, how = 'inner',
                   left_on='Unnamed: 0', right_on='Unnamed: 0', )


# concate with the GIANA PCAs
from sklearn.decomposition import PCA

pca = PCA(n_components=50, svd_solver='randomized', random_state=42)
GIANA_emb_PCA = pd.DataFrame(pca.fit_transform(GIANA_emb.iloc[:,2:]))
GIANA_emb_PCA = pd.concat([GIANA_emb.iloc[:,:2], GIANA_emb_PCA], axis=1)
# GIANA_emb_merged = pd.merge(GIANA_emb_PCA, cats, how = 'inner',
#                    left_on='Unnamed: 0', right_on='Unnamed: 0', )



#%%
fig, ax = plt.subplots(2,2, figsize=(8, 8))

tissue_cats = GIANA_emb_merged['tissue_y'].unique()
len_cats = GIANA_emb_merged['L_cdr3_beta'].unique()
vdjv_cats = GIANA_emb_merged['VDJ_1_v_call'].unique()

pca_num = 4

cluster_k = GIANA_emb_merged
vgene_cats = [grouped.keys()[0], grouped.keys()[1], grouped.keys()[-1], grouped.keys()[-2]]

i = 0

for subset in vgene_cats:
    v_subset = GIANA_emb_merged[GIANA_emb_merged['VDJ_1_v_call'] == subset]

    for category in tissue_cats:
        X_pca_sub = v_subset[v_subset['tissue_y'] == category].iloc[:, 2:18]

        ax[int(i/2), i%2].scatter(
            X_pca_sub.iloc[:,0], X_pca_sub.iloc[:, 1],
            label=category,
            alpha=0.8,
            linewidths=0.01,
        )
        ax[int(i/2), i%2].set_title(subset)
        
    i += 1
    
plt.legend()  # Add legend with title    
plt.show()











