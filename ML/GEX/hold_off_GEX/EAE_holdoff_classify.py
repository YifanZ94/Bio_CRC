# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:03:00 2025

# this script predict the 

@author: a4945
"""
import os
os.environ["SCIPY_ARRAY_API"] = "1"

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

Train_df = pd.read_csv("PCA50_train.csv")
Test_df = pd.read_csv("PCA50_2D2.csv")

num_features = 10
X_train = Train_df.iloc[:, 1:num_features+1]
X_test = Test_df.iloc[:, 1:num_features+1]

target_idx = 'tissue'
class_dict = {"CN":0, "SP":1}
Y_train = Train_df[target_idx].map(class_dict).astype('category')
Y_test = Test_df[target_idx].map(class_dict).astype('category')

#%%  resampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_train, Y_train = ros.fit_resample(X_train, Y_train)

from collections import Counter
print(sorted(Counter(Y_train).items()))

#%%
class_labels = Train_df[target_idx].astype("category").cat.categories
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
accuracy = (cm[0,0]+cm[1,1]) / sum(sum(cm))

disp.plot(cmap='Blues')

plt.title('acc:' + str(accuracy)[:5])
plt.show()
#%%  plot AUC 
# from sklearn.metrics import roc_curve, roc_auc_score
# import matplotlib.pyplot as plt

# # y_true: true binary labels (0 or 1)
# # y_scores: predicted probabilities for class 1 (NOT class labels)
# # e.g. from model.predict_proba(X)[:, 1]
# Y_test.iloc[-1] = 0

# fpr, tpr, thresholds = roc_curve(Y_test, test_pred[:,2])
# auc = roc_auc_score(Y_test, test_pred[:,2])

# plt.figure(figsize=(6, 5))
# plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

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
