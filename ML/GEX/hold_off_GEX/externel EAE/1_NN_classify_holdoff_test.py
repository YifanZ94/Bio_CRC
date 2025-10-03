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

seed=42
np.random.seed(seed)
tf.random.set_seed(seed)

import pandas as pd

#%%  load data
from sklearn.preprocessing import LabelEncoder

features = pd.read_csv("features_PCA50.csv", delimiter=",")

num_PCs = features.shape[1]-1

cat_ori = pd.read_csv("obs_classes.csv", delimiter=",")
cat_ori = cat_ori.fillna('sNaN')

ID_0516 = {'CMO301': '5_3', 'CMO302': '5_4', 'CMO303':'5_5', 
               'CMO304':'5_6', 'CMO305': '5_7', 'CMO317': '5_8',
               'CMO318': '5_3', 'CMO325':'5_4', 'CMO326':'5_5', 
               'CMO321':'5_6', 'CMO322': '5_7', 'CMO323': '5_8'}

ID_0605 = {'CMO301': '6_1', 'CMO302': '6_2', 'CMO303':'6_3', 'CMO304':'6_4',
           'CMO317': '6_1', 'CMO318':'6_2', 'CMO325':'6_3', 'CMO326':'6_4'}

cat_ori['date'] = cat_ori['date'].astype('str')
cat_ori['mouse_id'] = np.where(
    cat_ori['date'] == '605',
    cat_ori['mouse_id'].map(ID_0605),
    cat_ori['mouse_id'].map(ID_0516)
)

merged = pd.concat([features, cat_ori], axis=1, join = 'inner')
# left_on=0, right_on=0,

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

features = merged.iloc[:, 1:num_PCs+1]
cat = merged.iloc[:, num_PCs+1:]
features = pd.DataFrame(scaler.fit_transform(features))

df = pd.concat([features, cat], axis=1)

#%% hold mouses out as test
hold_off_id = ['6_4', '5_5']
test_idx = cat_ori['mouse_id'].isin(hold_off_id)
target_idx = 'tissue'
num_features = 10

df[target_idx] = df[target_idx].astype('category').cat.codes

X_train = df[~test_idx].iloc[:, :num_features]
X_test = df[test_idx].iloc[:, :num_features]

Y_train = df[~test_idx][target_idx]
Y_test = df[test_idx][target_idx]

#%%  resampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_train, Y_train = ros.fit_resample(X_train, Y_train)

from collections import Counter
print(sorted(Counter(Y_train).items()))

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
class_labels = cat_ori[target_idx].astype("category").cat.categories
num_classes = len(class_labels)

##   'softmax' --- categorical (one-hot encoded) 'categorical_crossentropy' --- 'CategoricalAccuracy'
##   'softmax' ---   (integers)  'sparse_categorical_crossentropy' ---  'accuracy' 

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
