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
    # Check each modality first (modality obs may have more rows than merged mdata.obs)
    sequences = None
    for mod_name in mdata.mod.keys():
        if column_name in mdata[mod_name].obs.columns:
            sequences = mdata[mod_name].obs[column_name]
            break
    
    # Fall back to top-level obs if not found in any modality
    if sequences is None:
        if column_name in mdata.obs.columns:
            sequences = mdata.obs[column_name]
        else:
            raise ValueError(f"Column '{column_name}' not found in mdata.obs or any modality obs")
    
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


from numpy.linalg import eigh
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class rCCA(BaseEstimator, TransformerMixin):
    """
    Ridge-regularized CCA (rCCA).
    Parameters
    ----------
    n_components : int
        Number of canonical pairs to extract.
    alpha_x : float
        L2 penalty added to S_xx (>=0).
    alpha_y : float
        L2 penalty added to S_yy (>=0).
    center : bool
        Center X and Y before fitting.
    scale : bool
        Z-score X and Y before fitting (recommended).
    """
    def __init__(self, n_components=10, alpha_x=1.0, alpha_y=1.0, center=True, scale=True):
        self.n_components = n_components
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.center = center
        self.scale = scale

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n = X.shape[0]
        if Y.shape[0] != n:
            raise ValueError("X and Y must have the same number of samples.")

        # preprocess
        self._scaler_x = StandardScaler(with_mean=self.center, with_std=self.scale).fit(X)
        self._scaler_y = StandardScaler(with_mean=self.center, with_std=self.scale).fit(Y)
        Xs = self._scaler_x.transform(X)
        Ys = self._scaler_y.transform(Y)

        # covariances
        Sxx = (Xs.T @ Xs) / (n - 1)
        Syy = (Ys.T @ Ys) / (n - 1)
        Sxy = (Xs.T @ Ys) / (n - 1)
        Syx = Sxy.T

        # ridge-regularize the auto-covariances
        p = Sxx.shape[0]
        q = Syy.shape[0]
        Sxx_reg = Sxx + self.alpha_x * np.eye(p)
        Syy_reg = Syy + self.alpha_y * np.eye(q)

        # Cholesky factors for stable solves
        Lx = cho_factor(Sxx_reg, lower=True, check_finite=False)
        Ly = cho_factor(Syy_reg, lower=True, check_finite=False)

        # Build the rCCA eigen-system: C = Sxx^{-1} Sxy Syy^{-1} Syx
        M = cho_solve(Lx, Sxy, check_finite=False)              # Sxx^{-1} Sxy
        C = M @ cho_solve(Ly, Syx, check_finite=False)          # Sxx^{-1} Sxy Syy^{-1} Syx

        # Solve for a; eigh since C is symmetric PSD
        eigvals, A = eigh(C)
        # sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        A = A[:, order]

        # keep leading components
        k = min(self.n_components, A.shape[1])
        A = A[:, :k]
        eigvals = np.clip(eigvals[:k], 0.0, None)
        # corresponding B from first-order condition
        B = cho_solve(Ly, Syx @ A, check_finite=False)
        # scale B so that correlations equal sqrt(eigvals)
        # (optional normalization below ensures unit variance in each view)
        for i in range(k):
            # normalize so that var(U_i)=var(V_i)=1
            ai = A[:, [i]]
            bi = B[:, [i]]
            denom_x = float(ai.T @ Sxx_reg @ ai)
            denom_y = float(bi.T @ Syy_reg @ bi)
            if denom_x > 0:
                A[:, [i]] /= np.sqrt(denom_x)
            if denom_y > 0:
                B[:, [i]] /= np.sqrt(denom_y)

        self.x_weights_ = A          # p x k
        self.y_weights_ = B          # q x k
        self.cancorr_ = np.sqrt(eigvals)  # canonical correlations (after ridge)

        # precompute to transform quickly
        self._fitted = True
        return self

    def transform(self, X, Y=None):
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Call fit before transform.")
        Xs = self._scaler_x.transform(X)
        U = Xs @ self.x_weights_
        if Y is None:
            return U
        Ys = self._scaler_y.transform(Y)
        V = Ys @ self.y_weights_
        return U, V

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X, Y)

    
def subset_fractions_in_CV_scores(n_component, mdata, abs_target, cutoff):
    all_results = []
    suffixes = []
    for i in range(n_component):
        cv_name = f'CV_score_{i}'

        # Positive / negative subsets
        mdata_sub_pos = mdata[mdata['gex'].obs[cv_name] > cutoff]
        mdata_sub_neg = mdata[mdata['gex'].obs[cv_name] < -cutoff]

        percentage_dict = {'CV_component': cv_name}

        # Helper to add percentages for a subset with a given label
        def add_percentages(subset, label):
            n = len(subset)
            if n == 0:
                return
            for col in abs_target:
                counts = subset['gex'].obs[col].value_counts()
                percentages = (counts / n * 100).round(2)
                for category, pct in percentages.items():
                    percentage_dict[f'{label}_{category}'] = pct

        # pos / neg fractions
        add_percentages(mdata_sub_pos, 'pos')
        add_percentages(mdata_sub_neg, 'neg')
        add_percentages(mdata, 'all')

        all_results.append(percentage_dict)

    df_CV = pd.DataFrame(all_results)

    # --- reorder columns: group pos_/neg_/all_ with the same suffix together ---
    cols = ['CV_component']    
    
    # breakpoint()
    
    suffixes = []
    for name in abs_target:
        suffixes += mdata['gex'].obs[name].astype('category').cat.categories.to_list()

    for suf in suffixes:
        for prefix in ('pos_', 'all_', 'neg_' ):
            name = f'{prefix}{suf}'
            if name in df_CV.columns:
                cols.append(name)

    df_CV = df_CV[cols]
    return df_CV

import seaborn as sns
def plot_subset_fractions_in_CV_scores(df_CV):    
    # --- reshape for plotting ---
    df_long = (
        df_CV.melt(id_vars="CV_component", var_name="Group", value_name="Percentage")
        .dropna(subset=["Percentage"])
    )

    # split "pos_X" -> label = pos/neg, category = X
    df_long["Label"] = df_long["Group"].str.split("_").str[0]
    df_long["Category"] = df_long["Group"].str.split("_").str[1]

    # --- plot ---
    fig, axs = plt.subplots(2,2, figsize=(10,8))
    for i in range(4):
        axi = axs[i//2, i%2]
        sns.barplot(
            data=df_long[df_long['CV_component']== 'CV_score_'+str(i)],
            x="Category",
            y="Percentage",
            hue="Label",
            errorbar=('pi', 90),
            ax = axi
        )
        axi.set_title('CV_score_'+str(i))

    plt.ylabel("Percentage (%)")
    plt.xlabel("Category")
    plt.legend(title="Subset", frameon=False)
    plt.tight_layout()
    plt.show()    
    
    
def train_test_corr(view_gene_c_train, view_tcr_c_train, view_gene_c_test, view_tcr_c_test):    
# Calculate the correlations separately for train and test sets
    correlations_train = []
    correlations_test = []
    cv_list = []

    print("Canonical Correlations Comparison:")
    print("="*60)
    print(f"{'Component':<15} {'Train Corr':<15} {'Test Corr':<15} {'Difference':<15}")
    print("="*60)

    for i in range(view_gene_c_train.shape[1]):
        # Calculate correlation for train set
        corr_train = np.corrcoef(view_gene_c_train[:, i], view_tcr_c_train[:, i])[0, 1]
        correlations_train.append(corr_train)

        # Calculate correlation for test set
        corr_test = np.corrcoef(view_gene_c_test[:, i], view_tcr_c_test[:, i])[0, 1]
        correlations_test.append(corr_test)

        # Calculate difference
        diff = corr_train - corr_test

        print(f"Component {i+1:<6} {corr_train:<15.4f} {corr_test:<15.4f} {diff:<15.4f}")
    
    return correlations_train, correlations_test

def plot_train_test_corr(correlations_train, correlations_test):    
    # Visualize correlation comparison between train and test set
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(correlations_train))

    # Plot dots for train and test
    ax.scatter(x, correlations_train, s=150, c='blue', alpha=0.7, label='Train', zorder=3)
    ax.scatter(x, correlations_test, s=150, c='orange', alpha=0.7, label='Test', zorder=3)

    # Connect train and test dots with lines
    for i in range(len(correlations_train)):
        ax.plot([i, i], [correlations_train[i], correlations_test[i]], 
                'k-', alpha=0.3, linewidth=1.5, zorder=2)

    ax.set_xlabel('CCA Component', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Train vs Test Canonical Correlations', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Component {i+1}' for i in range(len(correlations_train))])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    # ax.set_ylim(0, max(max(correlations_train), max(correlations_test)) * 1.1)

    # Add value labels
    for i, (train_val, test_val) in enumerate(zip(correlations_train, correlations_test)):
        ax.text(i, train_val, f'{train_val:.3f}', ha='center', va='bottom', fontsize=9, color='blue')
        ax.text(i, test_val, f'{test_val:.3f}', ha='center', va='top', fontsize=9, color='orange')

    plt.tight_layout()
    plt.show()