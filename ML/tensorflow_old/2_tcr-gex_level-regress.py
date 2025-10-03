# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 20:46:12 2025

@author: a4945
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
# gex = pd.read_csv("top50_Gex_PCA.csv", delimiter=",")

gex = pd.read_csv("selected_T_genes.csv", delimiter=",")

num_gex = gex.shape[1]-1

#%%   tcr embeddings
tcr_emb = pd.read_csv("TCR_AA_EncodingMatrix.txt", header = None, 
                      delimiter="\t")

tcr_emb = tcr_emb.iloc[:,1:]   # remove the AA seqeunce
col_name = ["BC"]
for i in range(tcr_emb.shape[1]-1):
    col_name.append("TCR_EB" + str(i+1))
tcr_emb.columns = col_name

num_EB = tcr_emb.shape[1]-1   

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# tcr_scaler = MinMaxScaler()
# gex_scaler = MinMaxScaler()

tcr_scaler = StandardScaler()
gex_scaler = StandardScaler()


tcr_emb.iloc[:,1:] = tcr_scaler.fit_transform(tcr_emb.iloc[:,1:])
gex.iloc[:,1:] = gex_scaler.fit_transform(gex.iloc[:,1:])

merged = tcr_emb.merge(gex, left_on="BC", right_on='Unnamed: 0', how = 'inner')
merged.drop(columns=['BC', 'Unnamed: 0'], inplace=True)

#%%  DF to torch dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

generator = torch.Generator(device= device)
target_col_idx = [0]
target_col_idx = [x + num_EB for x in target_col_idx]

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
        self.targets = dataframe.iloc[:, target_column].values.astype(np.float32)
        
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


all_set = DataFrameDataset(merged, target_col_idx)

train_set, test_set = torch.utils.data.random_split(all_set, [.85, .15], 
                                                    generator=torch.Generator(device=device))

train_dataloader = DataLoader(train_set, batch_size=128, shuffle=False, generator=torch.Generator(device=device))

train_features, train_labels = next(iter(train_dataloader))

test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True, generator=torch.Generator(device=device))

#%%
# Initialize the model

class NeuralNet(nn.Module):
    def __init__(self, num_features, num_labels):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)  # Hidden layer with 32 units
        self.fc2 = nn.Linear(64, 32)            # Hidden layer with 16 units
        self.fc3 = nn.Linear(32, 16)            # Hidden layer with 16 units
        self.fc_end = nn.Linear(16, num_labels)   # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc_end(x))
        
        return x

def train(model, device, dataloader, optimizer, criterion, 
          n_epoch):
    
    torch.set_grad_enabled(True)
    model.train()
    loss_epoch = []
    
    for iter in range(1, n_epoch + 1):
        epoch_loss = 0
        print(f"Epoch {iter}\n-------------------------------")
        # model.zero_grad() # clear the gradients
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            output = model(X)
            
            # batch_loss = criterion(output, y)
            # epoch_loss += batch_loss.item()
            # # optimize parameters
            # batch_loss.backward()
            
            
            # nn.utils.clip_grad_norm_(model.parameters(), 1)
            for name, param in model.named_parameters():
                if batch%50 == 0 and param.grad is not None:
                    print(f"{name} gradient norm: {param.grad.norm().item()}")     
                    
            optimizer.step()
            optimizer.zero_grad()
            
        loss_epoch.append(epoch_loss/(batch+1))

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
            X, y = X.to(device), y.to(device)
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
    

#%%
model = NeuralNet(num_EB, len(target_col_idx)).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr= 1e-3)

#%%  model checkpoint load
# checkpoint = torch.load('model_checkpoint_800.pth', weights_only=True)

# model.load_state_dict(checkpoint['model_state_dict'])  # Restore model weights
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Restore optimizer
# start_epoch = checkpoint['epoch']  # Get the last trained epoch
# loss = checkpoint['loss']  # Restore loss (optional)


#%%  main    
epochs = 1
import time
start = time.time()

train_loss = train(model, device, train_dataloader, optimizer, criterion, n_epoch = epochs)
end = time.time()
print(f"the training used {end-start} seconds" )


# model = torch.load('trained_model.pt', weights_only=False)

pred, target, loss = test(test_dataloader, model, criterion)   
#  loss per column
test_loss_column = ((pred - target) ** 2).mean(dim=0)
# pred_real = all_set.gex_scaler.inverse_transform(pred.to('cpu'))
# ref_real = all_set.gex_scaler.inverse_transform(target.to('cpu'))

# rmse = np.sqrt(np.mean((pred_real - ref_real) ** 2))

#%%
import matplotlib.pyplot as plt
x = np.arange(epochs)
plt.plot(x, train_loss)
plt.xlabel('epoch')

# plt.xticks(range(x[0], x[-1]+1))

plt.ylabel('loss')
plt.show()

#%%
for batch, (X, y) in enumerate(test_dataloader):
    X, y = X.to(device), y.to(device)

# for name, param in model.named_parameters():
#     print(f"{name} gradient norm: {param.grad.norm().item()}")

#%%  SAVE checkpoint

# torch.save({
#     'epoch': 50,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': train_loss[-1]
# }, 'model_checkpoint_1300.pth')


#%%  save model
# torch.save(model.state_dict(),'trained_model.pt')
