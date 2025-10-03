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

#%%  load data
from sklearn.preprocessing import LabelEncoder

features = pd.read_csv("features_PCA50.csv", delimiter=",")

num_PCs = features.shape[1]-1
num_features = 10

cat_ori = pd.read_csv("obs_classes.csv", delimiter=",")
cat_ori = cat_ori.fillna('sNaN')

merged = pd.concat([features, cat_ori], axis=1, join = 'inner')
# left_on=0, right_on=0,

features = merged.iloc[:, 1:num_PCs+1]
cat = merged.iloc[:, num_PCs+1:]

for col in cat_ori.columns:
    cat[col] = cat_ori[col].astype('category').cat.codes

df = pd.concat([features, cat], axis=1)

target_idx = 3

#%%  feature distribution and selection
# import seaborn as sns
# import matplotlib.pyplot as plt

# def IQR_filter(data):
#     q1 = np.percentile(data, 25)
#     q3 = np.percentile(data, 75)
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
    
#     return data[(data >= lower_bound) & (data <= upper_bound)]

# k = 6

# from scipy.stats import ks_2samp


# for feature_dim in range(k):
#     classes = list(set(cat_ori.iloc[:,target_idx]))
#     pcs_distribution = {}
    
#     for i in range(len(classes)):
#         idx = cat_ori.iloc[:,target_idx] == classes[i]
#         pcs_distribution[f'component_{i}'] = features.iloc[:,feature_dim][idx]
        
#         pcs_distribution[f'component_{i}'] = IQR_filter(pcs_distribution[f'component_{i}'])
        
#         sns.kdeplot(pcs_distribution[f'component_{i}'], label=classes[i], fill=True)
        
   
#     stat, p_value = ks_2samp(pcs_distribution['component_0'], pcs_distribution['component_1'])
#     if p_value < 0.05:
#         print(f'{feature_dim}th PCA component is different')
                
#         plt.legend()
#         plt.xlabel('Value')
#         plt.ylabel('Density')
#         plt.title(f'{feature_dim}th PCA component')
#         plt.show()

#%% shuffle to mix LEE data and EXternal data

# only Keep the LEE lab data
# idx = 19843
# df = df[:idx]

# shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# tran_percent = 0.7
# idx = round(tran_percent*len(df))
# idx2 = 22940

# X_train = shuffled_df.iloc[:idx, :num_features]
# X_test = shuffled_df.iloc[idx:, :num_features]

# Y_train = shuffled_df.iloc[:idx, num_features + target_idx]
# Y_test = shuffled_df.iloc[idx:, num_features + target_idx]

#%% NOT shuffle: Use LEE data as training and External set as test
# idx = 19843

idx = cat_ori['tissue'].isin(("CN","SP")) 

shuffled_df = df

X_train = shuffled_df[idx].iloc[:,:num_features]
Y_train = shuffled_df[idx].iloc[:, num_PCs + target_idx]

X_test = shuffled_df[~idx].iloc[:, :num_features]
Y_test = shuffled_df[~idx].iloc[:, num_PCs + target_idx]

#%%  resampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)

from collections import Counter
print(sorted(Counter(Y_resampled).items()))

X_train = X_resampled
Y_train = Y_resampled

#%% pre processing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%  Binary classes
# Define the model
# num_classes = Y_train.value_counts().shape[0]

# model = Sequential([
#     Dense(32, activation='relu', input_shape=(num_features,)),  # Hidden layer with 64 units
#     Dense(16, activation='relu'),                              # Hidden layer with 32 units
#     Dense(1, activation='sigmoid')                             # Output layer for binary classification
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='BinaryCrossentropy',
#               metrics=[tf.keras.metrics.BinaryAccuracy()])

# # Train the model
# model.fit(X_train, Y_train, epochs=3, batch_size=32)
# validation_data=(X_val, y_val)

#%%
num_classes = cat_ori.iloc[:,target_idx].value_counts().shape[0]

##   'softmax' --- categorical (one-hot encoded) 'categorical_crossentropy' --- 'CategoricalAccuracy'
##   'softmax' ---   (not one-hot)  'sparse_categorical_crossentropy' ---  'accuracy' 

model = Sequential([
    Dense(32, activation='relu', input_shape=(num_features,)),  # Hidden layer with 64 units
    Dense(16, activation='relu'),                              # Hidden layer with 32 units
    Dense(num_classes, activation='softmax')                   # Output layer for classification
])

# Compile the model (one-hot encoded)
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['CategoricalAccuracy'])

# learning_rate = 0.001  # You can choose your desired learning rate
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0) # Example with clipnorm
# Compile the model NOT (one-hot encoded)
model.compile(optimizer='adam',
              # optimizer= optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
model.fit(X_train, Y_train, epochs=5, batch_size=32)


#%%  test
test_pred = model.predict(X_test)

#%%  sigmoid
# error = 0
# for i in range(len(test_pred)):
#     test_pred[i] = 1 if test_pred[i] >= 0.5 else 0
#     if test_pred[i] != Y_test.iloc[i]:
#         error += 1 

#%%  soft max
class_pred = []

for i in range(len(test_pred)):
    class_pred.append(np.argmax(test_pred[i]))

#%%      plot confusion matrix  
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(Y_test, class_pred)
s = cat_ori.iloc[:, target_idx].astype("category")
class_labels = s.cat.categories

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels)

disp.plot(cmap='Blues')
plt.show()

#%%  plot AUC 
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# y_true: true binary labels (0 or 1)
# y_scores: predicted probabilities for class 1 (NOT class labels)
# e.g. from model.predict_proba(X)[:, 1]
Y_test.iloc[-1] = 0

fpr, tpr, thresholds = roc_curve(Y_test, test_pred[:,2])
auc = roc_auc_score(Y_test, test_pred[:,2])

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



#%% PCA plot
import matplotlib.pyplot as plt

# Plot the first two PCA components
figure1 = plt.figure(figsize=(8, 8))

X_pca = features
y = cat_ori.iloc[:,target_idx].unique()

for category in y:
    X_pca_sub = X_pca[cat_ori['tissue'] == category]
    print(X_pca_sub.shape)
    
    plt.figure(figure1)

    plt.scatter(
        X_pca_sub.iloc[:,0], X_pca_sub.iloc[:, 1],
        label=category,
        alpha=0.7,
        edgecolor='k'
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA - First Two Components")
plt.legend(title="Tissue")  # Add legend with title
plt.grid(True)
plt.tight_layout()
plt.show()

#%%   SHAP
import shap
X100 = shap.utils.sample(X_train, 100)

explainer = shap.Explainer(model, X100)
shap_values = explainer(X_train[:200,:])

shap.summary_plot(shap_values, X_train)

#%% resampled features
# figure1 = plt.figure(figsize=(8, 8))
# y = y_resampled.unique()

# for category in y:
#     X_pca_sub = X_resampled[y_resampled == category]
#     print(X_pca_sub.shape)
    
#     plt.figure(figure1)

#     plt.scatter(
#         X_pca_sub.iloc[:,0], X_pca_sub.iloc[:, 1],
#         label=category,
#         alpha=0.7,
#         edgecolor='k'
#     )

# plt.legend(title="Tissue")  # Add legend with title
# plt.grid(True)
# plt.tight_layout()
# plt.show()
