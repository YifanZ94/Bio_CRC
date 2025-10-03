# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:03:00 2025

# this script predict the 

@author: a4945
"""

from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed

set_seed(455)

import numpy as np
import tensorflow as tf
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
np.random.seed(455)


#%%  load data
from sklearn.preprocessing import LabelEncoder

features_train = pd.read_csv("features_train.csv", delimiter=",")
features_test = pd.read_csv("features_test.csv", delimiter=",")
labels_train = pd.read_csv("labels_train.csv", delimiter=",")
labels_test = pd.read_csv("labels_test.csv", delimiter=",")

num_pca = features_train.shape[1]-1
num_features = 5
target_idx = 2

labels_train = labels_train.fillna('sNaN')
labels_train = labels_train.iloc[:, 1:6]
labels_test = labels_test.fillna('sNaN')
labels_test = labels_test.iloc[:, 1:6]

labels_all = pd.concat([labels_train, labels_test], axis=0)
train_idx = labels_all['tissue'].isin(("CN","SP")) 

s = labels_all.iloc[:, target_idx].astype("category")
class_labels = s.cat.categories

encoder = LabelEncoder()

for col in labels_train.columns:
    # cat[col] = cat_ori[col].astype('category').cat.codes
    labels_all[col] = encoder.fit_transform(labels_all[col].astype('str'))
    

num_classes = labels_all.iloc[:,target_idx].value_counts().shape[0]

#%%  feature selection by distribution

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
# idx = 46314
# df = df[:idx]

# shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# tran_percent = 0.7
# idx = round(tran_percent*len(df))
# idx2 = 22940

# X_train = shuffled_df.iloc[:idx, :num_features]
# X_test = shuffled_df.iloc[idx:, :num_features]

# Y_train = shuffled_df.iloc[:idx, num_pca + target_idx]
# Y_test = shuffled_df.iloc[idx:, num_pca + target_idx]

#%% NOT shuffle: Use LEE data as training and External set as test

X_train = features_train.iloc[:, 1:num_features+1]
Y_train = labels_all[train_idx].iloc[:, target_idx]

X_test = features_test.iloc[:, 1:num_features+1]
Y_test = labels_all[~train_idx].iloc[:, target_idx]
   
 
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


#%%  confusion plot      
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(Y_test, class_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels)

disp.plot(cmap='Blues')
plt.show()


#%% PCA plot
# import matplotlib.pyplot as plt

# # Plot the first two PCA components
# figure1 = plt.figure(figsize=(8, 8))

# X_pca = features
# y = cat_ori.iloc[:,target_idx].unique()

# for category in y:
#     X_pca_sub = X_pca[cat_ori['tissue'] == category]
#     print(X_pca_sub.shape)
    
#     plt.figure(figure1)

#     plt.scatter(
#         X_pca_sub.iloc[:,0], X_pca_sub.iloc[:, 1],
#         label=category,
#         alpha=0.7,
#         edgecolor='k'
#     )

# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.title("PCA - First Two Components")
# plt.legend(title="Tissue")  # Add legend with title
# plt.grid(True)
# plt.tight_layout()
# plt.show()
