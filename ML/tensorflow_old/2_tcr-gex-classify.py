# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:51:15 2025

@author: a4945
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Check if CUDA is available
# device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda')

torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

#%%   gex 
gex = pd.read_csv("selected_T_genes.csv", delimiter=",")

# gex = pd.read_csv("top50_Gex_PCA.csv", delimiter=",")

num_gex = gex.shape[1]-1

#%%   tcr embeddings
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

tcr_emb = pd.read_csv("TCR_AA_EncodingMatrix.txt", header = None, 
                      delimiter="\t")
tcr_emb = tcr_emb.iloc[:,1:]
col_name = ["BC"]
for i in range(tcr_emb.shape[1]-1):
    col_name.append("TCR_EB" + str(i+1))
    
tcr_emb.columns = col_name
num_EB = tcr_emb.shape[1]-1   

tcr_scaler = MinMaxScaler()
tcr_emb.iloc[:,1:] = tcr_scaler.fit_transform(tcr_emb.iloc[:,1:])

merged = tcr_emb.merge(gex, right_on='Unnamed: 0', left_on="BC", how = 'inner')
merged.drop(columns=['BC', 'Unnamed: 0'], inplace=True)

#  scalar value to integer classes
gene_level_discrete = merged.iloc[:,num_EB:]

le = LabelEncoder()
column_list = gene_level_discrete.columns.tolist()

num_of_target_classes = []

for col in column_list:
    gene_level_discrete[col] = le.fit_transform(gene_level_discrete[col]) 
    num_of_target_classes.append(max(gene_level_discrete[col])+1)
    
merged.iloc[:,num_EB:] = gene_level_discrete


#%%  DF to torch dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler

generator = torch.Generator(device= device)


class DataFrameDataset(Dataset):
    def __init__(self, dataframe, target_column):
        """
        Args:
            dataframe (pd.DataFrame): The input dataframe.
            target_column (str, optional): The name of the target column. If None, dataset returns only features.
        """
        self.dataframe = dataframe
        self.target_column = target_column
        self.features = dataframe.iloc[:, :num_EB].values.astype(np.float32)
        
        target_col = [x + num_EB for x in target_column]
        self.targets = dataframe.iloc[:, target_col].values.astype(np.float32)
        
        # self.tcr_scaler = MinMaxScaler()
        # self.gex_scaler = MinMaxScaler()
        # self.features = self.tcr_scaler.fit_transform(self.features)
        # self.targets = self.gex_scaler.fit_transform(self.targets)
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        x = torch.tensor(self.features[index, :])
        if self.targets is not None:
            y = torch.tensor(self.targets[index])
            return x, y
        return x


target_col_index = [0]

all_set = DataFrameDataset(merged, target_col_index)

train_set, test_set = torch.utils.data.random_split(all_set, [.85, .15], 
                                                    generator=torch.Generator(device=device))

train_dataloader = DataLoader(train_set, batch_size=128, shuffle=False, generator=torch.Generator(device=device))

train_features, train_labels = next(iter(train_dataloader))

test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True, generator=torch.Generator(device=device))

#%%
# Initialize the model

class NeuralNet(nn.Module):
    def __init__(self, num_features, num_classes_list):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 32)  # Hidden layer with 32 units
        self.fc2 = nn.Linear(32, 16)            # Hidden layer with 16 units
        # self.fc3 = nn.Linear(16, 1)   # Output layer
        
        self.output_heads = nn.ModuleList([
            nn.Linear(16, num_classes) for num_classes in num_classes_list])
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = []
        
        for i, target_i in enumerate(self.output_heads):
            output.append(F.softmax(target_i(x), dim=1))
        
        # x = F.relu(self.fc3(x))  # Apply softmax along the correct dimension
        
        return output


def train(model, device, dataloader, optimizer, criterion, 
          n_epoch):
    
    torch.set_grad_enabled(True)
    model.train()
    loss_epoch = []
    
    for iter in range(1, n_epoch + 1):
        print(f"Epoch {iter}\n-------------------------------")
        # nn.zero_grad() # clear the gradients
        
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            output = model(X)
            batch_loss = 0
            
            for i, out_i in enumerate(output):
                batch_loss += criterion(out_i, y[:,i]) 

            # optimize parameters
            batch_loss.backward()
            
            for name, param in model.named_parameters():
                if batch%100 == 0 and param.grad is not None:
                    print(f"{name} gradient norm: {param.grad.norm().item()}")
            
            # nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            
        loss_epoch.append(batch_loss.item())
        
    return loss_epoch


def test(dataloader, model, loss_fn):
    print('test start')
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    pred_scalar = None
    target_scaler =  None
    
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred_i = model(X)
            test_loss += loss_fn(pred_i, y).item()

            if pred_scalar is not None:
                pred_scalar = torch.cat((pred_scalar, pred_i), dim=0)  # Concatenate along rows
                target_scaler = torch.cat((target_scaler, y), dim=0)
            else:
                pred_scalar = pred_i  # Assign if it's the first time
                target_scaler = y
                
    test_loss /= num_batches
    print(f" Avg loss: {test_loss:>8f} \n")
    return pred_scalar, target_scaler, test_loss
    

#%%  main    
target_classes = [num_of_target_classes[x] for x in target_col_index]

model = NeuralNet(num_EB, target_classes).to(device)
criterion = nn.NLLLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.01)


#%% Define the loss function and optimizer

epochs = 1
import time
start = time.time()

train_loss = train(model, device, train_dataloader, optimizer, criterion, n_epoch = epochs)
end = time.time()
print(f"the training used {end-start} seconds" )

pred, target, loss = test(test_dataloader, model, criterion)   

# pred_real = all_set.gex_scaler.inverse_transform(pred.to('cpu'))
# ref_real = all_set.gex_scaler.inverse_transform(target.to('cpu'))

# rmse = np.sqrt(np.mean((pred_real - ref_real) ** 2))

#%%
import matplotlib.pyplot as plt
plt.plot(train_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


