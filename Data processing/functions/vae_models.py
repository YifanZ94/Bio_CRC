"""
Variational Autoencoder (VAE) Models for multi-modal and single-modal data.

This module provides VAE implementations including:
- MultiModalConditionalVAE: Standard multi-modal VAE
- TransformerMultiModalConditionalVAE: Multi-modal VAE with transformer encoders
- SingleModalConditionalVAE: Single-modal VAE
"""

import torch
import torch.nn as nn

# Import transformer encoders
try:
    from .transformer_encoders import TransformerEncoder, FeatureTransformerEncoder
except ImportError:
    # Fallback for direct import
    from transformer_encoders import TransformerEncoder, FeatureTransformerEncoder


class BaseConditionalVAE(nn.Module):
    """
    Base class for Conditional VAE models with common functionality.
    """
    def reparameterize(self, mu, logvar):
        """Reparameterization trick - shared across all VAE models"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class MultiModalConditionalVAE(BaseConditionalVAE):
    """
    Multi-modal Conditional VAE Model.
    
    Processes both TCR embeddings and GEX data together with conditional information.
    
    Args:
        tcr_dim: Dimension of TCR embeddings
        gex_dim: Dimension of GEX data
        condition_dim: Dimension of conditional input (e.g., sample ID one-hot)
        latent_dim: Dimension of latent space (default: 128)
        hidden_dim: Dimension of hidden layers (default: 512)
        n_classes: Number of classification classes (default: 3)
    """
    def __init__(self, tcr_dim, gex_dim, condition_dim, latent_dim=128, hidden_dim=512, n_classes=3):
        super(MultiModalConditionalVAE, self).__init__()
        
        self.tcr_dim = tcr_dim
        self.gex_dim = gex_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        # Encoder: combine both modalities with condition
        encoder_input_dim = tcr_dim + gex_dim + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder: reconstruct both modalities
        decoder_input_dim = latent_dim + condition_dim
        self.decoder_tcr = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, tcr_dim)
        )
        
        self.decoder_gex = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, gex_dim)
        )
        
        # Classifier head for tissue prediction
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def encode(self, tcr, gex, condition):
        """Encode inputs to latent space"""
        x = torch.cat([tcr, gex, condition], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z, condition):
        """Decode from latent space"""
        z_cond = torch.cat([z, condition], dim=1)
        tcr_recon = self.decoder_tcr(z_cond)
        gex_recon = self.decoder_gex(z_cond)
        return tcr_recon, gex_recon
    
    def forward(self, tcr, gex, condition):
        """Forward pass"""
        mu, logvar = self.encode(tcr, gex, condition)
        z = self.reparameterize(mu, logvar)
        tcr_recon, gex_recon = self.decode(z, condition)
        tissue_pred = self.classifier(z)
        return tcr_recon, gex_recon, mu, logvar, z, tissue_pred


class TransformerMultiModalConditionalVAE(BaseConditionalVAE):
    """
    Multi-modal VAE with transformer encoders for TCR and GEX data.
    
    Uses self-attention transformers to process each modality separately before VAE encoding.
    
    Args:
        tcr_dim: Dimension of TCR embeddings
        gex_dim: Dimension of GEX data
        condition_dim: Dimension of conditional input
        latent_dim: Dimension of latent space (default: 128)
        hidden_dim: Dimension of hidden layers (default: 512)
        n_classes: Number of classification classes (default: 3)
        transformer_d_model: Dimension of transformer embeddings (default: 256)
        transformer_nhead: Number of attention heads (default: 8)
        transformer_num_layers: Number of transformer layers (default: 2)
        transformer_dim_feedforward: Feedforward dimension (default: 1024)
        transformer_dropout: Dropout rate for transformers (default: 0.1)
        use_chunked_transformer: Whether to use chunked transformer (default: True)
        chunk_size: Size of each chunk for FeatureTransformerEncoder (default: 64)
    """
    def __init__(self, tcr_dim, gex_dim, condition_dim, latent_dim=128, hidden_dim=512, 
                 n_classes=3, transformer_d_model=256, transformer_nhead=8, 
                 transformer_num_layers=2, transformer_dim_feedforward=1024, 
                 transformer_dropout=0.1, use_chunked_transformer=True, chunk_size=64):
        super(TransformerMultiModalConditionalVAE, self).__init__()
        
        self.tcr_dim = tcr_dim
        self.gex_dim = gex_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        # Transformer encoders for each modality
        if use_chunked_transformer:
            self.tcr_transformer = FeatureTransformerEncoder(
                input_dim=tcr_dim,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
                chunk_size=chunk_size
            )
            self.gex_transformer = FeatureTransformerEncoder(
                input_dim=gex_dim,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout,
                chunk_size=chunk_size
            )
        else:
            self.tcr_transformer = TransformerEncoder(
                input_dim=tcr_dim,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout
            )
            self.gex_transformer = TransformerEncoder(
                input_dim=gex_dim,
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dim_feedforward=transformer_dim_feedforward,
                dropout=transformer_dropout
            )
        
        # VAE Encoder: combine transformer outputs with condition
        encoder_input_dim = transformer_d_model + transformer_d_model + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder: reconstruct both modalities
        decoder_input_dim = latent_dim + condition_dim
        self.decoder_tcr = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, tcr_dim)
        )
        
        self.decoder_gex = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, gex_dim)
        )
        
        # Classifier head for tissue prediction
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def encode(self, tcr, gex, condition):
        """Encode inputs to latent space using transformer encoders"""
        # Process each modality through its transformer
        tcr_transformed = self.tcr_transformer(tcr)  # (batch_size, transformer_d_model)
        gex_transformed = self.gex_transformer(gex)  # (batch_size, transformer_d_model)
        
        # Combine transformer outputs with condition
        x = torch.cat([tcr_transformed, gex_transformed, condition], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z, condition):
        """Decode from latent space"""
        z_cond = torch.cat([z, condition], dim=1)
        tcr_recon = self.decoder_tcr(z_cond)
        gex_recon = self.decoder_gex(z_cond)
        return tcr_recon, gex_recon
    
    def forward(self, tcr, gex, condition):
        """Forward pass"""
        mu, logvar = self.encode(tcr, gex, condition)
        z = self.reparameterize(mu, logvar)
        tcr_recon, gex_recon = self.decode(z, condition)
        tissue_pred = self.classifier(z)
        return tcr_recon, gex_recon, mu, logvar, z, tissue_pred


class SingleModalConditionalVAE(BaseConditionalVAE):
    """
    Single-modality Conditional VAE Model (for GEX-only or TCR-only).
    
    Args:
        input_dim: Dimension of input modality
        condition_dim: Dimension of conditional input
        latent_dim: Dimension of latent space (default: 128)
        hidden_dim: Dimension of hidden layers (default: 512)
        n_classes: Number of classification classes (default: 3)
    """
    def __init__(self, input_dim, condition_dim, latent_dim=128, hidden_dim=512, n_classes=3):
        super(SingleModalConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        # Encoder: input modality + condition
        encoder_input_dim = input_dim + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder: reconstruct input modality
        decoder_input_dim = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Classifier head for tissue prediction
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def encode(self, x, condition):
        """Encode inputs to latent space"""
        x_cond = torch.cat([x, condition], dim=1)
        h = self.encoder(x_cond)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z, condition):
        """Decode from latent space"""
        z_cond = torch.cat([z, condition], dim=1)
        x_recon = self.decoder(z_cond)
        return x_recon
    
    def forward(self, x, condition):
        """Forward pass"""
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, condition)
        tissue_pred = self.classifier(z)
        return x_recon, mu, logvar, z, tissue_pred

