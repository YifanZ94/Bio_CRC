# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:24:07 2025

@author: a4945
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:03:00 2025

# this script predict the 

@author: a4945
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["SCIPY_ARRAY_API"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import KFold
torch.manual_seed(455)

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(455)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

#%% load deepTCR embeding
tcr_rep = pd.read_csv("../TCR/tcr_features.csv", delimiter=",")
# tcr_rep = tcr_rep.dropna()

TCRdist_MOG = pd.read_csv("../TCR/TCRdist_MOG.csv", delimiter=",")
TCRdist_MOG['cdr3_b_aa'] = TCRdist_MOG['cdr3_b_aa'].str[1:-1]
TCRdist_MOG = TCRdist_MOG[['cdr3_b_aa', 'TCRdist_MOG']]

#%%
T_states = pd.read_csv("../TCR/T_states.csv", delimiter=",")
#### error in data labeling   ####
T_states['clone_id_size'] = T_states['clone_id_size'].astype('int')
T_states['state'].replace({"Effector": 'Naive/Mem', "Mem": 'Naive/Mem'}, inplace = True) 

tcr_rep = tcr_rep.merge(T_states, left_on='cell_id', right_on = 'Unnamed: 0', how ='inner')
tcr_rep = tcr_rep.merge(TCRdist_MOG, left_on='VDJ_1_cdr3_aa', right_on = 'cdr3_b_aa', how ='inner')

#%%  load DeepTCR embedings
matrix = pd.read_csv('../TCR/DTCRU_extracted_features_96.csv', sep = ',')
matrix.drop(columns={'Label'}, inplace=True)   # un useful col
matrix['CDR3_Beta'] = matrix['CDR3_Beta'].str[1:-1]    # remove first C and last F in AA

merged = pd.merge(tcr_rep, matrix, how='inner', left_on='VDJ_1_cdr3_aa', right_on='CDR3_Beta')
merged = merged[(merged['VDJ_1_v_call'] == merged['V_Beta']) & 
                (merged['VDJ_1_j_call'] == merged['J_Beta'])]

#%% process the gex features
cat_gex = pd.read_csv("../TCR/gex_obs_classes.csv", delimiter=",")   # all 0605 data
cat_gex = cat_gex.iloc[:,0:6]
cat_gex = cat_gex.dropna()
cat_gex.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)

cat_gex['date'] = cat_gex['cell_id'].str.split('_').str[1]

ID_0516 = {'CMO301': '5_3', 'CMO302': '5_4', 'CMO303':'5_5', 
               'CMO304':'5_6', 'CMO305': '5_7', 'CMO317': '5_8',
               'CMO318': '5_3', 'CMO325':'5_4', 'CMO326':'5_5', 
               'CMO321':'5_6', 'CMO322': '5_7', 'CMO323': '5_8'}

ID_0605 = {'CMO301': '6_1', 'CMO302': '6_2', 'CMO303':'6_3', 'CMO304':'6_4',
           'CMO317': '6_1', 'CMO318':'6_2', 'CMO325':'6_3', 'CMO326':'6_4'}

cat_gex['mouse_id'] = np.where(
    cat_gex['date'] == '0605',
    cat_gex['mouse_id'].map(ID_0605),
    cat_gex['mouse_id'].map(ID_0516)
)

dup_cols = merged.columns.intersection(cat_gex.columns)
cat_gex_clean = cat_gex.drop(columns=dup_cols)    
df_all_features = pd.merge(cat_gex, merged, how='inner', on='cell_id')
df_all_features.drop(columns='tissue_y', inplace=True)
df_all_features.rename(columns={'tissue_x':'tissue'}, inplace=True)

df_all_features['mouse_id'] = df_all_features['mouse_id'].astype('category')
mouse_id_cats = df_all_features['mouse_id'].astype('category').cat.categories

#%% load top 10 chemokine genes
chemo_profile = pd.read_csv('../TCR/top10_chemok.csv')
df_all_features = df_all_features.merge(chemo_profile, left_on='cell_id', right_on='Unnamed: 0')

#%%  subsets
df_all_features['is_cloned'] = df_all_features.duplicated('CDR3_Beta', keep=False)
df_all_features['two_sites'] = df_all_features.groupby('CDR3_Beta')['tissue'].transform(lambda x: x.nunique() > 1)

#%% df_clone_twoLoc analysis
# clone_twoLoc_sorted = df_clone_twoLoc.assign(dup_count=df_clone_twoLoc.groupby('CDR3_Beta')['CDR3_Beta'].transform('count')) \
#               .sort_values(by='dup_count', ascending=False) \
#               .drop_duplicates(subset='CDR3_Beta')
              
# # Count tissue occurrences within each clonotype
# counts = df_clone_twoLoc.groupby(['CDR3_Beta', 'tissue']).size().unstack(fill_value=0)
# ratios = counts.div(counts.sum(axis=1), axis=0)

# clone_twoLoc_sorted = clone_twoLoc_sorted.merge(ratios, how ='inner', left_on = 'CDR3_Beta', right_on = 'CDR3_Beta')
# clone_twoLoc_sorted.to_csv('paired_clon_twoLoc.csv')

# plt.figure(figsize=(6, 4))
# plt.scatter(clone_twoLoc_sorted['CN'], clone_twoLoc_sorted['dup_count'])
# plt.xlabel('CNS Ratio')
# plt.ylabel('Clonotype Count')
# plt.grid(True)
# plt.show()


#%% data pre-processing
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocessing(df_in, target):
    str_cols = df_in.select_dtypes(include=["object", "string", "category"]).columns

    # One-hot encode those, keep numeric columns as-is
    df = pd.get_dummies(df_in, columns = str_cols, dtype="uint8", dummy_na=True)   
    df.columns = df.columns.astype(str)
    
    feature_names = df.columns
    
    # resampling
    ros = RandomOverSampler(random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(df, target)

    return X_resampled, Y_resampled, feature_names

#%% build model
class TCRClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TCRClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(16, num_classes)
        
        # L1 and L2 regularization equivalent
        self.l1_l2_reg = 0.01
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = self.dropout3(x)
        x = F.softmax(self.output(x), dim=1)
        return x
    
    def l1_l2_loss(self):
        l1_loss = sum(torch.norm(p, 1) for p in self.parameters())
        l2_loss = sum(torch.norm(p, 2) for p in self.parameters())
        return self.l1_l2_reg * (l1_loss + l2_loss)

def build_model(input_size, num_classes):
    model = TCRClassifier(input_size, num_classes)
    return model

#%%      plot confusion matrix  
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion(Y_test, class_pred, s, title):
    cm = confusion_matrix(Y_test, class_pred)
    s = merged[target_class].astype('category')
    class_labels = s.cat.categories
    pred_accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels)
    disp.plot(cmap='Blues')
    plt.title(str(title) + "  acc:" + str(round(pred_accuracy, 3)))
    plt.savefig(str(title) + "_confusion.png")
    plt.show()

## plot AUC 
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def plot_ROC(Y_test, test_pred, title):
    Y_test.iloc[-1] = 0
    # y_true: true binary labels (0 or 1)
    # y_scores: predicted probabilities for class 1 (NOT class labels)
    # e.g. from model.predict_proba(X)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(Y_test, test_pred[:,1])
    auc = roc_auc_score(Y_test, test_pred[:,1])
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(title))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(title) + "_ROC.png")
    plt.show()

#%%  train
# select features
input_cat_features = ['manual_cell_type', 'state', 'clone_id_size', 'TCRdist_MOG' ]   #    

input_embs = [str(s) for s in range(94)]
chemo_keys = chemo_profile.columns.values[1:].tolist()
epoch_num = 100

# check cols
# df_all_features.columns
# subset
df_all_features = df_all_features[df_all_features['TCRdist_MOG'] < 100]

Bool_list = [True]
Cell_types = ['CD4+ T', 'CD8+ T', 'Treg']
#     , 'Treg'

target_class = 'tissue'

for ind1 in Bool_list:
    M_sub1 = df_all_features[df_all_features['is_cloned'] == ind1]
    # M_sub1 = df_all_features                     # Not subsetting
    
    for ind2 in Cell_types:
        M_sub = M_sub1[M_sub1['manual_cell_type'] == ind2]
        # M_sub = M_sub1                          # Not subsetting
        
        mouse_id = str(ind1) + ind2        
        features = M_sub[input_cat_features + input_embs + chemo_keys]      # 
        
        num_classes = merged[target_class].astype('category').value_counts().shape[0]
        target = M_sub[target_class].astype('category').cat.codes
        s = M_sub[target_class]
        
        test_id = ['6_4', '5_5']
        test_idx = M_sub['mouse_id'].isin(test_id)
        
        features_train = features[~test_idx]
        target_train = target[~test_idx]
        X_train, Y_train, _ = preprocessing(features_train, target_train)
        
        features_test = features[test_idx]
        target_test = target[test_idx]
        X_test, Y_test, feature_names = preprocessing(features_test, target_test)
        
        num_features = X_train.shape[1]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        Y_train_tensor = torch.LongTensor(Y_train.values)
        X_test_tensor = torch.FloatTensor(X_test.values)
        Y_test_tensor = torch.LongTensor(Y_test.values)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Build and train model
        model = build_model(num_features, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Training loop
        model.train()
        for epoch in range(epoch_num):
            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y) + model.l1_l2_loss()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        #% test
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_tensor).numpy()
        class_pred = np.argmax(test_pred, axis=1)
        
        plot_confusion(Y_test, class_pred, s, mouse_id)
        plot_ROC(Y_test, test_pred, mouse_id)

#%%  save model
# torch.save(model.state_dict(), 'TCR.pth')
           
# model = build_model(num_features, num_classes)
# model.load_state_dict(torch.load('TCR.pth'))

#%% shap explain
# explain all the predictions in the test set

# import shap
# def shap_eavl(X_train, X_test, features):
    
    # Background (masker) — sample to keep things fast and stable
rng = np.random.default_rng(0)
bg_idx = rng.choice(X_train.shape[0], size=min(100, X_train.shape[0]), replace=False)
background = X_train.iloc[bg_idx]

# Prediction function that includes preprocessing if you want to explain raw X
# Here we already precomputed X_train_s/X_test_s; if you'd rather pass raw X to SHAP,
# define: f = lambda data: model.predict(scaler.transform(data), verbose=0)
def predict_function(data):
    data_tensor = torch.FloatTensor(data.values)
    model.eval()
    with torch.no_grad():
        return model(data_tensor).numpy()

f = predict_function

# Create the explainer (auto picks a fast, gradient-based method for TF/Keras when possible)
explainer = shap.Explainer(f, shap.maskers.Independent(background))

# Use a manageable slice for speed (e.g., 500 samples)
sample_idx = rng.choice(X_test.shape[0], size=min(30, X_test.shape[0]), replace=False)
X_eval = X_test.iloc[sample_idx]

# Compute explanations
shap_values = explainer(X_eval)  # returns a shap.Explanation

shap_values.feature_names = feature_names.tolist()

# k = 0  # or np.argmax(model.predict(X_eval), axis=1)[i] for per-sample class
# shap.plots.beeswarm(shap_values[:, :, k], max_display=5)        # class k

# or overall ranking across classes:
shap.plots.bar(shap_values.abs.mean(axis=2), max_display=20)     # mean|SHAP| over classes

# shap_eavl(X_train, X_test, feature_names)
