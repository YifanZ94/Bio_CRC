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

idx_train = cat_ori['tissue'].isin(("CN","SP")) 

# unify the test set labels
# print(set(cat_ori['tissue']))
# s_dict = pd.Series({'CN': 'CNS', 'SP': 'Spleen', 'CNS_ext_2':'CNS'})
# cat_ori['tissue'] = cat_ori['tissue'].replace(s_dict)
# true_labels = list(set(cat_ori['tissue']))

merged = pd.concat([features, cat_ori], axis=1, join = 'inner')
features = merged.iloc[:, 1:num_PCs+1]
cat = merged.iloc[:, num_PCs+1:]

for col in cat_ori.columns:
    cat[col] = cat_ori[col].astype('category').cat.codes

df = pd.concat([features, cat], axis=1)

target_idx = 3


#%% NOT shuffle: Use LEE data as training and External set as test
# idx = 19843

shuffled_df = df

X_train = shuffled_df[idx_train].iloc[:,:num_features]
Y_train = shuffled_df[idx_train].iloc[:, num_PCs + target_idx]

X_test = shuffled_df[~idx_train].iloc[:, :num_features]
Y_test = shuffled_df[~idx_train].iloc[:, num_PCs + target_idx]

num_classes = Y_train.value_counts().shape[0]
# num_classes = cat_ori.iloc[:,target_idx].value_counts().shape[0]

#%%  resampling
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(random_state=0)
# X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)

# from collections import Counter
# print(sorted(Counter(Y_resampled).items()))

# X_train = X_resampled
# Y_train = Y_resampled

#%% pre processing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.kernel_approximation import RBFSampler

rbf = RBFSampler(gamma=1.0, n_components=500, random_state=0)
X_train_rff = rbf.fit_transform(X_train)
X_test_rff = rbf.transform(X_test)
feature_dim = X_train_rff.shape[1]

#%% scikit learn SVM
from sklearn import svm
clf = svm.SVC(degree= 5, )
clf.fit(X_train_rff, Y_train)

y_pred = clf.predict(X_test_rff)

from collections import Counter
print(sorted(Counter(y_pred).items()))

from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(Y_test, y_pred))
# print(classification_report(Y_test, y_pred))

#%%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(Y_test, y_pred)
s = cat_ori.iloc[:, target_idx].astype("category")
labels = s.cat.categories

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# , display_labels = true_labels
disp.plot(cmap='Blues', colorbar=False)

# Customize axis tick labels manually
disp.ax_.set_xticklabels([''] + labels)
disp.ax_.set_yticklabels([''] + labels)

plt.show()


#%%  SVM from scratch
# X = tf.constant(X_train, dtype=tf.float32)
# y = tf.constant(Y_train, dtype=tf.float32)

# # Initialize model parameters
# W = tf.Variable(tf.random.normal([2, 1]))
# b = tf.Variable(tf.zeros([1]))

# # Hyperparameters
# C = 1.0   # Regularization strength (soft margin)
# lr = 0.01
# epochs = 200

# # Convert to tensors
# X_train_tf = tf.constant(X_train_rff, dtype=tf.float32)
# labels = []
# Ws, bs = [], []

# loss_class = []

# for c in range(2):
#     y_c = np.where(Y_train == c, 1.0, -1.0)
#     y_tf = tf.constant(y_c.reshape(-1,1), dtype=tf.float32)
#     W = tf.Variable(tf.random.normal([feature_dim,1]))
#     b = tf.Variable(tf.zeros([1]))
#     optimizer = tf.keras.optimizers.SGD(lr)
#     loss_c = []
    
#     for epoch in range(epochs):
#         with tf.GradientTape() as tape:
#             logits = tf.matmul(X_train_tf, W) + b
#             hinge = tf.reduce_mean(tf.maximum(0., 1 - y_tf * logits))
#             loss = 0.5 * tf.reduce_sum(W**2) + C * hinge
#         loss_c.append(loss.numpy())
#         grads = tape.gradient(loss, [W, b])
#         optimizer.apply_gradients(zip(grads, [W, b]))
        
#     Ws.append(W); bs.append(b); loss_class.append(loss_c)

# test
# X_test_tf = tf.constant(X_test_rff, dtype=tf.float32)
# # Compute score for each class: shape (n_samples, num_classes)
# scores = tf.concat([tf.matmul(X_test_tf, W) + b for W,b in zip(Ws, bs)], axis=1)
# y_pred = tf.argmax(scores, axis=1).numpy()

# from sklearn.metrics import accuracy_score, classification_report
# print("Accuracy:", accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test, y_pred))

# from collections import Counter
# print(sorted(Counter(y_pred).items()))