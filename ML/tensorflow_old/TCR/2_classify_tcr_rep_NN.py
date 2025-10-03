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
from sklearn.preprocessing import LabelEncoder

#%%  load GIANA data
# features = pd.read_csv("GIANA_EncodingMatrix.txt", sep = '\t', header = None)
# features.rename(columns={0:'cdr3_b_aa', 1: 'BC'}, inplace=True)

# cat_ori = pd.read_csv("gex_obs_classes.csv", delimiter=",")
# cat_ori = cat_ori.fillna('sNaN')

# # select cell type sub-clusters
# # sub_cluster = cat_ori['mouse_id'].isin(['CMO317']) & cat_ori['manual_cell_type'].isin(['CD4+ T'])
# # cat_ori = cat_ori[sub_cluster]

# merged = pd.merge(features, cat_ori, left_on='BC', right_on='Unnamed: 0', how = 'inner')

# first_emb_idx = 2
# num_features = 94
# target_class = 'tissue'
# features = merged.iloc[:, first_emb_idx:first_emb_idx+num_features+1]
# target = merged[target_class].astype('category').cat.codes

# df = pd.concat([features, target], axis=1)


#%% load deepTCR embeding
tcr_rep = pd.read_csv("tcr_features.csv", delimiter=",")
tcr_rep = tcr_rep.dropna()

matrix = pd.read_csv('DTCRU_extracted_features_96.csv', sep = ',')
matrix.drop(columns={'Label'}, inplace=True)   # un useful col
matrix['CDR3_Beta'] = matrix['CDR3_Beta'].str[1:-1]    # remove first C and last F in AA

merged = pd.merge(tcr_rep, matrix, how='inner', left_on='VDJ_1_cdr3_aa', right_on='CDR3_Beta')
merged = merged[(merged['VDJ_1_v_call'] == merged['V_Beta']) & 
                (merged['VDJ_1_j_call'] == merged['J_Beta'])]

#%% process the gex features
cat_gex = pd.read_csv("gex_obs_classes.csv", delimiter=",")   # all 0605 data
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

#%%
# Step 1: Keep only duplicates in CDR3_Beta
is_cloned = df_all_features.duplicated('CDR3_Beta', keep=False)
df_cloned = df_all_features[is_cloned]
df_single = df_all_features[~is_cloned]

# Step 2: From those, keep only CDR3_Beta with more than 2 tissue classes
mask = df_cloned.groupby('CDR3_Beta')['tissue'].transform(lambda x: x.nunique() > 1)
df_twoSource = df_cloned[mask]

#%%
# select between df_paired or df_single
M = df_cloned

target_class = 'tissue'

# subset on mouse
M_sub0 = M[M['mouse_id'] == mouse_id_cats[0]]

M_sub = M_sub0[M_sub0['manual_cell_type'] == 'Treg']

class_counts = []
class_counts.append(M_sub[target_class].value_counts().to_dict())

#%%    
first_emb_idx = M.columns.get_loc('0')
num_features = 94

features = M_sub.iloc[:, first_emb_idx:first_emb_idx+num_features+1]
target = M_sub[target_class].astype('category').cat.codes

# print the target classes
M_sub[target_class].astype('category').value_counts()

df = pd.concat([features, target], axis=1)

#%% shuffle to mix LEE data and EXternal data

shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
tran_percent = 0.7
idx = round(tran_percent*len(df))

X_train = shuffled_df.iloc[:idx, :num_features]
X_test = shuffled_df.iloc[idx:, :num_features]

Y_train = shuffled_df.iloc[:idx, -1]
Y_test = shuffled_df.iloc[idx:, -1]

#%%  resampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)
X_test, Y_test = ros.fit_resample(X_test, Y_test)

from collections import Counter
print(sorted(Counter(Y_resampled).items()))

X_train = X_resampled
Y_train = Y_resampled

#%% pre processing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
num_classes = merged[target_class].astype('category').value_counts().shape[0]
##   'softmax' --- categorical (one-hot encoded) 'categorical_crossentropy' --- 'CategoricalAccuracy'
##   'softmax' ---   (not one-hot)  'sparse_categorical_crossentropy' ---  'accuracy' 
from tensorflow.keras.regularizers import l1_l2

model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),  # Hidden layer with 64 units
    Dropout(0.3),
    Dense(32, activation='relu'),  
    Dropout(0.2),
    Dense(16, activation='relu', ),    
    Dropout(0.2),                          # Hidden layer with 32 units
    Dense(num_classes, activation='softmax', kernel_regularizer=l1_l2(0.01))                   # Output layer for classification
])

# Compile the model (one-hot encoded)
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['CategoricalAccuracy'])

# learning_rate = 0.001  # You can choose your desired learning rate
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0) # Example with clipnorm

# Compile the model NOT (one-hot encoded)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=200, batch_size=32)

#%%  sigmoid
# error = 0
# for i in range(len(test_pred)):
#     test_pred[i] = 1 if test_pred[i] >= 0.5 else 0
#     if test_pred[i] != Y_test.iloc[i]:
#         error += 1 

#%%  test
test_pred = model.predict(X_test)
class_pred = []

for i in range(len(test_pred)):
    class_pred.append(np.argmax(test_pred[i]))

#%%      plot confusion matrix  
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(Y_test, class_pred)
s = merged[target_class].astype('category')
class_labels = s.cat.categories

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels)
disp.plot(cmap='Blues')
plt.show()

pred_accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
print("The test accuracty is : " + str(pred_accuracy))

## plot AUC 
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# y_true: true binary labels (0 or 1)
# y_scores: predicted probabilities for class 1 (NOT class labels)
# e.g. from model.predict_proba(X)[:, 1]
Y_test.iloc[-1] = 0

fpr, tpr, thresholds = roc_curve(Y_test, test_pred[:,1])
auc = roc_auc_score(Y_test, test_pred[:,1])

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%  save model
# model.save('TCR.keras')
           
# from tensorflow.keras.models import load_model
# model_load = load_model('TCR.keras')

