import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import muon as mu
import scanpy as sc
import scirpy as ir
np.random.seed(42)
import random
random.seed(42)


def encode_tcr_sequence(seq, max_length, atchley_factors):
    """
    Encode a TCR amino acid sequence into a vector using Atchley factors.
    
    Parameters:
    - seq: amino acid sequence string
    - max_length: maximum sequence length for padding
    - atchley_factors: dictionary mapping amino acids to their Atchley factors
    
    Returns:
    - numpy array of shape ((max_length+1)*5,)
    """
    vector = []
    
    # Handle None or NaN values
    if pd.isna(seq) or seq is None:
        return np.zeros((max_length + 1) * 5)
    
    # Convert sequence to uppercase
    seq = str(seq).upper()
    
    # Encode each amino acid
    for aa in seq:
        if aa in atchley_factors:
            vector.extend(atchley_factors[aa])
        else:
            # Unknown amino acid, use zeros
            vector.extend([0, 0, 0, 0, 0])
    
    # Pad with zeros to reach (max_length+1)*5
    current_length = len(vector)
    target_length = (max_length + 1) * 5
    vector.extend([0] * (target_length - current_length))
    
    return np.array(vector)

def vectorize_tcr_column(mdata, column_name, atchley_factors):
    """
    Vectorize a TCR amino acid column in mdata.obs using Atchley factors.
    
    Parameters:
    - mdata: MuData object
    - column_name: name of the column containing TCR sequences
    - atchley_factors: dictionary mapping amino acids to their Atchley factors
    
    Returns:
    - numpy array of shape (n_obs, (L+1)*5) where L is max sequence length
    """
    # Get the column data
    sequences = mdata.obs[column_name]
    
    # Find maximum length L
    max_length = mdata.obs[column_name].str.len().max()

    # Encode all sequences
    encoded_vectors = []
    for seq in sequences:
        vec = encode_tcr_sequence(seq, max_length, atchley_factors)
        encoded_vectors.append(vec)
    
    return np.array(encoded_vectors)


def compute_aa_composition(seq):
    """
    Compute the percentage composition of each of the 20 amino acids.
    
    Parameters:
    - seq: amino acid sequence string
    
    Returns:
    - numpy array of shape (20,) with percentages for each AA
    """
    # Standard 20 amino acids in alphabetical order
    standard_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Handle None or NaN values
    if pd.isna(seq) or seq is None:
        return np.zeros(20)
    
    # Convert to uppercase and count
    seq = str(seq).upper()
    seq_len = len(seq)
    
    if seq_len == 0:
        return np.zeros(20)
    
    # Count each amino acid
    aa_counts = np.zeros(20)
    for i, aa in enumerate(standard_aas):
        aa_counts[i] = seq.count(aa)
    
    # Convert to percentages
    aa_percentages = aa_counts / seq_len
    
    return aa_percentages

def compute_aa_composition_matrix(mdata, column_name):
    """
    Compute AA composition for all sequences in a column.
    
    Parameters:
    - mdata: MuData object
    - column_name: name of the column containing TCR sequences
    
    Returns:
    - numpy array of shape (n_obs, 20) with AA composition percentages
    """
    sequences = mdata.obs[column_name]
    composition_matrix = np.array([compute_aa_composition(seq) for seq in sequences])
    return composition_matrix

def compute_sequence_lengths(mdata, column_name):
    """
    Compute the length of each sequence in a column.
    
    Parameters:
    - mdata: MuData object
    - column_name: name of the column containing TCR sequences
    
    Returns:
    - numpy array of sequence lengths
    """
    sequences = mdata.obs[column_name]
    lengths = np.array([len(str(seq)) if pd.notna(seq) and seq is not None else 0 for seq in sequences])
    return lengths


def onehot_encode_categorical(mdata, column_name):
    """
    One-hot encode a categorical column with an additional dimension for unknown/missing values.
    
    Parameters:
    - mdata: MuData object
    - column_name: name of the column to encode
    
    Returns:
    - numpy array of shape (n_obs, n_categories + 1) where +1 is for unknown/missing
    """
    # Get the column data
    data = mdata.obs[column_name]
    
    # Get unique non-null categories
    categories = sorted([x for x in data.unique() if pd.notna(x)])
    n_categories = len(categories)
    
    print(f"{column_name}: {n_categories} unique categories + 1 unknown = {n_categories + 1} dimensions")
    
    # Create category to index mapping
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # Initialize one-hot matrix (last column is for unknown)
    onehot_matrix = np.zeros((len(data), n_categories + 1))
    
    # Fill in one-hot encoding
    for i, value in enumerate(data):
        if pd.isna(value) or value not in cat_to_idx:
            # Unknown/missing value goes in the last column
            onehot_matrix[i, -1] = 1
        else:
            # Known category
            onehot_matrix[i, cat_to_idx[value]] = 1
    
    return onehot_matrix

