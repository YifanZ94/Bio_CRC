# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:03:00 2025

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

features = pd.read_csv("features.csv", delimiter=",")
num_features = features.shape[1]-1

cat_ori = pd.read_csv("cats_cluster.csv", delimiter=",")
cat_ori = cat_ori.fillna('sNaN')

merged = features.merge(cat_ori, left_on='Unnamed: 0', right_on='Unnamed: 0', how = 'inner')

features = merged.iloc[:, 1:num_features+1]
cat = merged.iloc[:, num_features+1:]

le = LabelEncoder()
column_list = cat.columns.tolist()
num_of_classes = []

for col in column_list:
    cat[col] = le.fit_transform(cat[col]) 
    num_of_classes.append(max(cat[col])+1)

df = pd.concat([features, cat], axis=1)
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)


#%% pre processing
tran_percent = 0.7
target_idx = 4
idx = round(tran_percent*len(df))
num_classes = num_of_classes[target_idx]


X_train = shuffled_df.iloc[:idx, :num_features]
X_test = shuffled_df.iloc[idx:, :num_features]

Y_train_index = shuffled_df.iloc[:idx, num_features + target_idx]
Y_test_index = shuffled_df.iloc[idx:, num_features + target_idx]

Y_train = tf.one_hot(Y_train_index, depth=num_classes)
Y_test = tf.one_hot(Y_test_index, depth=num_classes)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%

##   'softmax' --- categorical (one-hot encoded) 'categorical_crossentropy' --- 'CategoricalAccuracy'
##   'softmax' ---   (not one-hot)  'sparse_categorical_crossentropy' ---  'accuracy' 

model = Sequential([
    Dense(32, activation='relu', input_shape=(num_features,)),  # Hidden layer with 64 units
    Dense(16, activation='relu'),                              # Hidden layer with 32 units
    Dense(num_classes, activation='softmax')                             # Output layer for binary classification
])

# Compile the model (one-hot encoded)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['CategoricalAccuracy'])


# Compile the model NOT (one-hot encoded)
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32)


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


#%%        
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(Y_test_index, class_pred)
s = cat_ori.iloc[:, 1+ target_idx].astype("category")
class_labels = s.cat.categories

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels)

disp.plot(cmap='Blues')
plt.show()





