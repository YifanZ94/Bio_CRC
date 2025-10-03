# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 12:31:46 2025

@author: a4945
"""
import sys

def custom_excepthook(exc_type, exc_value, exc_traceback):
    print(f"{exc_type.__name__}: {exc_value}")

sys.excepthook = custom_excepthook

import pandas as pd

all_embs = pd.read_csv('DTCRU_features_all.csv')
all_obs = pd.read_csv('tcr_obs_all.csv')
sub_2D2_embs = pd.read_csv('2D2_deepTCR_embs.csv')

all_embs['CDR3_Beta'] = all_embs['CDR3_Beta'].str[1:-1]

all_obs['V_Beta'] = all_obs['VDJ_1_v_call'].str.split('+').str[0]
all_obs['J_Beta'] = all_obs['VDJ_1_j_call'].str.split('+').str[0]

sub_2D2_embs['V_Beta'] = sub_2D2_embs['v_b_gene'].str.split('*').str[0]
sub_2D2_embs['J_Beta'] = sub_2D2_embs['j_b_gene'].str.split('*').str[0]

pd_temp = pd.merge(all_embs, all_obs[['VDJ_1_cdr3_aa','V_Beta','J_Beta', 'tissue']],
                   left_on = ['CDR3_Beta','V_Beta','J_Beta'],
                   right_on=['VDJ_1_cdr3_aa','V_Beta','J_Beta'], how ='inner')

#%%
idx_2D2 = pd_temp['CDR3_Beta'].isin(sub_2D2_embs['cdr3_b_aa'])

# df1.to_csv('DTCRU_extracted_features_96.csv', index=False)
