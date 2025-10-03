#!/usr/bin/env python3
"""
Script to run the EAE TCR-GEX analysis with a single unified model
"""

import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
os.environ["SCIPY_ARRAY_API"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import KFold
torch.manual_seed(455)

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(455)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score, accuracy_score

def setup_test_environment():
    """
    Function to input test name and create output directory
    """
    # Generate default name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = f"EAE_single_model_{timestamp}"
    
    # Create output directory
    output_dir = f"results_{test_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Test name: {test_name}")
    print(f"Output directory: {output_dir}")
    print(f"All results will be saved to: {os.path.abspath(output_dir)}")
    
    return test_name, output_dir

def preprocessing(df_in, target):
    str_cols = df_in.select_dtypes(include=["object", "string", "category"]).columns

    # One-hot encode those, keep numeric columns as-is
    df = pd.get_dummies(df_in, columns = str_cols, dtype="uint8", dummy_na=True)   
    df.columns = df.columns.astype(str)
    
    feature_names = df.columns
    
    # resampling
    ros = RandomOverSampler(random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(df, target)

    return X_resampled, Y_resampled, feature_names

class TCRClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TCRClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(16, num_classes)
        
        # L1 and L2 regularization equivalent
        self.l1_l2_reg = 0.01
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.relu(self.layer3(x))
        x = self.dropout3(x)
        x = F.softmax(self.output(x), dim=1)
        return x
    
    def l1_l2_loss(self):
        l1_loss = sum(torch.norm(p, 1) for p in self.parameters())
        l2_loss = sum(torch.norm(p, 2) for p in self.parameters())
        return self.l1_l2_reg * (l1_loss + l2_loss)

def build_model(input_size, num_classes):
    model = TCRClassifier(input_size, num_classes)
    return model

def plot_confusion(Y_test, class_pred, title, output_dir):
    cm = confusion_matrix(Y_test, class_pred)
    class_labels = ['CN', 'SP']  # Based on tissue values
    pred_accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels)
    disp.plot(cmap='Blues')
    plt.title(str(title) + "  acc:" + str(round(pred_accuracy, 3)))
    
    # Save to output directory
    output_path = os.path.join(output_dir, f"{title}_confusion.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()  # Close to free memory

def plot_ROC(Y_test, test_pred, title, output_dir):
    fpr, tpr, thresholds = roc_curve(Y_test, test_pred[:,1])
    auc = roc_auc_score(Y_test, test_pred[:,1])
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(title))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save to output directory
    output_path = os.path.join(output_dir, f"{title}_ROC.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to: {output_path}")
    plt.close()  # Close to free memory

def main():
    print("Starting EAE TCR-GEX Analysis with Single Model...")
    
    # Setup test environment
    TEST_NAME, OUTPUT_DIR = setup_test_environment()
    
    # Load EAE dataset
    print("Loading EAE_allTcells.csv...")
    df_all_features = pd.read_csv('EAE_allTcells.csv')
    
    print(f"Dataset shape: {df_all_features.shape}")
    print(f"Tissue distribution: {df_all_features['tissue'].value_counts()}")
    print(f"Mouse ID distribution: {df_all_features['mouse_id'].value_counts()}")
    
    # Training configuration - SIMPLIFIED
    # Only use clone_id_size and TCRdist_MOG (exclude modified_cell_type)
    input_cat_features = ['clone_id_size', 'TCRdist_MOG']
    
    # Use only first 20 Temb embeddings (instead of 96)
    input_embs = [f'Temb_{s}' for s in range(20)]
    
    # Gene expression features
    gene_features = ['gene_Cd4', 'gene_Cd8a', 'gene_Cd8b1', 'gene_Nkg7', 'gene_Foxp3', 
                     'gene_Ikzf2', 'gene_Ctla4', 'gene_Il2ra', 'gene_Ccr6', 'gene_Il22']
    
    epoch_num = 50  # Reduced from 100
    
    # Filter data
    df_all_features = df_all_features[df_all_features['TCRdist_MOG'] < 200]
    
    target_class = 'tissue'
    
    print(f"Filtered dataset shape: {df_all_features.shape}")
    print(f"Tissue distribution after filtering: {df_all_features['tissue'].value_counts()}")
    print(f"Features used: {len(input_cat_features + input_embs + gene_features)} total")
    print(f"  - Categorical features: {input_cat_features}")
    print(f"  - Temb embeddings: {len(input_embs)} (first 20)")
    print(f"  - Gene features: {len(gene_features)}")
    
    # Save dataset summary
    summary_file = os.path.join(OUTPUT_DIR, f"{TEST_NAME}_dataset_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Test Name: {TEST_NAME}\n")
        f.write(f"Dataset shape: {df_all_features.shape}\n")
        f.write(f"Tissue distribution:\n{df_all_features['tissue'].value_counts()}\n")
        f.write(f"Mouse ID distribution:\n{df_all_features['mouse_id'].value_counts()}\n")
        f.write(f"Features used: {len(input_cat_features + input_embs + gene_features)}\n")
        f.write(f"Categorical features: {input_cat_features}\n")
        f.write(f"Temb embeddings: {len(input_embs)} (first 20)\n")
        f.write(f"Gene features: {len(gene_features)}\n")
    print(f"Dataset summary saved to: {summary_file}")
    
    # SINGLE MODEL TRAINING - Use entire training set
    print("\n=== Training Single Unified Model ===")
    
    # Get all features for the entire dataset
    features = df_all_features[input_cat_features + input_embs + gene_features]
    
    # Get target
    target = df_all_features[target_class].astype('category').cat.codes
    s = df_all_features[target_class]
    
    # Train/test split: mouse_id '5_3' and '5_4' as test group
    test_id = ['5_3', '5_4']
    test_idx = df_all_features['mouse_id'].isin(test_id)
    
    print(f"Train samples: {(~test_idx).sum()}")
    print(f"Test samples: {test_idx.sum()}")
    print(f"Test mouse distribution: {df_all_features[test_idx]['mouse_id'].value_counts()}")
    
    # Split data
    features_train = features[~test_idx]
    target_train = target[~test_idx]
    X_train, Y_train, _ = preprocessing(features_train, target_train)
    
    features_test = features[test_idx]
    target_test = target[test_idx]
    X_test, Y_test, feature_names = preprocessing(features_test, target_test)
    
    num_features = X_train.shape[1]
    num_classes = df_all_features[target_class].astype('category').value_counts().shape[0]
    
    print(f"Training unified model: {X_train.shape[0]} train, {X_test.shape[0]} test samples, {num_features} features")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    Y_train_tensor = torch.LongTensor(Y_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    Y_test_tensor = torch.LongTensor(Y_test.values)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Build and train model
    model = build_model(num_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training loop
    print("Training model...")
    model.train()
    for epoch in range(epoch_num):
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y) + model.l1_l2_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epoch_num} completed")
    
    # Test
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor).numpy()
    class_pred = np.argmax(test_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(Y_test, class_pred)
    fpr, tpr, _ = roc_curve(Y_test, test_pred[:,1])
    auc = roc_auc_score(Y_test, test_pred[:,1])
    
    print(f"Results for unified model:")
    print(f"  Accuracy = {accuracy:.3f}")
    print(f"  AUC = {auc:.3f}")
    
    # Create plots
    plot_confusion(Y_test, class_pred, "unified_model", OUTPUT_DIR)
    plot_ROC(Y_test, test_pred, "unified_model", OUTPUT_DIR)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"unified_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Save results summary
    results_summary = {
        'model_name': 'unified_model',
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'num_features': num_features,
        'accuracy': accuracy,
        'auc': auc
    }
    
    results_df = pd.DataFrame([results_summary])
    results_file = os.path.join(OUTPUT_DIR, f"{TEST_NAME}_results_summary.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results summary saved to: {results_file}")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
