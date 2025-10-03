# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:13:45 2025

@author: a4945
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Check if CUDA is available
# device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')

torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")

#%%  load data
from sklearn.preprocessing import LabelEncoder
import pandas as pd

features = pd.read_csv("top50_Gex_PCA.csv", delimiter=",")
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

#%%  DF to torch dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
generator = torch.Generator(device= device)

class DataFrameDataset(Dataset):
    def __init__(self, dataframe, target_column=None):
        """
        Args:
            dataframe (pd.DataFrame): The input dataframe.
            target_column (str, optional): The name of the target column. If None, dataset returns only features.
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.features = dataframe.iloc[:, :num_features].values.astype(np.float32)

        # Convert to NumPy for fast access
        if target_column != None:
            self.targets = dataframe.iloc[:, num_features+target_column].values
        else:
            self.targets = None

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        x = torch.tensor(self.features[index, :])
        if self.targets is not None:
            y = torch.tensor(self.targets[index]).long()
            # y = F.one_hot(y, num_classes = class_i[self.target_column])
            return x, y
        return x

# , generator=generator, device="cpu"

#%%  NN
class NeuralNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 32)  # Hidden layer with 32 units
        self.fc2 = nn.Linear(32, 16)            # Hidden layer with 16 units
        self.fc3 = nn.Linear(16, num_classes)   # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)  # Apply softmax along the correct dimension
        return x


#%%  data prepare

target_idx = 1
all_set = DataFrameDataset(df, target_column=target_idx)

train_set, test_set = torch.utils.data.random_split(all_set, [.85, .15], 
                                                    generator=torch.Generator(device=device))

train_dataloader = DataLoader(train_set, batch_size=128, shuffle=False, generator=torch.Generator(device=device))

train_features, train_labels = next(iter(train_dataloader))

test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True, generator=torch.Generator(device=device))


#%%
# Initialize the model

model = NeuralNet(num_features, num_of_classes[target_idx]).to(device)

# Define the loss function and optimizer
criterion = nn.NLLLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(nn, device, dataloader, optimizer, criterion, 
          n_epoch, report_every = 5):
    
    nn.to(device)
    nn.train()
    
    for iter in range(1, n_epoch + 1):
        print(f"Epoch {iter}\n-------------------------------")
        nn.zero_grad() # clear the gradients
        current_loss = 0
        
        for batch, (X, y) in enumerate(dataloader):
            # X, y = X.to(device), y.to(device)
            
            output = nn(X)
            batch_loss = criterion(output, y)

            # optimize parameters
            batch_loss.backward()
            
            # nn.utils.clip_grad_norm_(nn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 50 == 0:
                print(batch_loss.item())

            current_loss += batch_loss.item() / (batch+1)

        # all_losses.append(current_loss / (batch+1) )
        
        # if iter % report_every == 0:
        #     print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        # current_loss = 0


def test(dataloader, model, loss_fn):
    print('test start')
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    pred_class = None
    target_class =  None
    
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            top_n, top_i = pred.topk(1)

            if pred_class is not None:
                pred_class = torch.cat((pred_class, top_i), dim=0)  # Concatenate along rows
                target_class = torch.cat((target_class, y), dim=0)
            else:
                pred_class = top_i  # Assign if it's the first time
                target_class = y
                
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return pred_class, target_class
    

#%%  main    
epochs = 10
import time
start = time.time()

train(model, device, train_dataloader, optimizer, criterion, n_epoch = epochs, report_every = 50)
end = time.time()
print(f"the training used {end-start} seconds" )

pred, target = test(test_dataloader, model, criterion)   
 

#%%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(target.to(torch.device('cpu')), pred.to(torch.device('cpu')))

s = cat_ori.iloc[:, 1+ target_idx].astype("category")

class_labels = s.cat.categories

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels)

disp.plot(cmap='Blues')
    
# ax = plt.gca()  # Get current axes
# ax.set_xticklabels(class_labels, rotation=45)
# ax.set_yticklabels(ax.get_yticks(), rotation=30)

plt.show()


#%%
# device_gpu = torch.device('cuda')

# for batch, (X, y) in enumerate(test_dataloader):
#     print(batch)
#     start = time.time()
#     X, y = X.to(device), y.to(device)
#     end = time.time()
#     print(end-start)
#     batch_loss = 0    

# out = model(X)
# top_n, top_i = out.topk(1)    

