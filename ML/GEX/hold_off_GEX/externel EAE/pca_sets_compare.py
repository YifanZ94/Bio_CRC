# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:21:22 2025

@author: a4945
"""

import numpy as np
import pandas as pd


features_1 = pd.read_csv("features_PCA50.csv", delimiter=",")
features_2 = pd.read_csv("features_PCA50_3.csv", delimiter=",")
idx = 46314
features_1 = features_1.iloc[:idx,1:]
features_2 = features_2.iloc[:idx,1:]

# umap_1 = pd.read_csv("features_Umap2_seed42_rerun.csv", delimiter=",")
# umap_2 = pd.read_csv("features_Umap2_3.csv", delimiter=",")
# umap_1 = umap_1.iloc[:idx,1:]
# umap_2 = umap_2.iloc[:idx,1:]


classes_1 = pd.read_csv("obs_classes.csv", delimiter=",")
classes_2 = pd.read_csv("obs_classes_3.csv", delimiter=",")
classes_1 = classes_1.iloc[:idx,1:6]
classes_2 = classes_2.iloc[:idx,1:6]

for col in classes_1.columns:
    classes_1[col] = classes_1[col].astype('category').cat.codes
for col in classes_2.columns:
    classes_2[col] = classes_2[col].astype('category').cat.codes


#%%
feature_diff = features_1 - features_2
print(feature_diff.max())

class_diff = classes_1 - classes_2
print(class_diff.max())