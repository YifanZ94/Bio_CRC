"""
Variational Autoencoder (VAE) Models for multi-modal and single-modal data.

This module provides VAE implementations including:
- MultiModalConditionalVAE: Standard multi-modal VAE
- TransformerMultiModalConditionalVAE: Multi-modal VAE with transformer encoders
- SingleModalConditionalVAE: Single-modal VAE
"""

import torch
import torch.nn as nn
import math

class BaseConditionalVAE(nn.Module):
    """
    Base class for Conditional VAE models with common functionality.
    """
    def reparameterize(self, mu, logvar):
        """Reparameterization trick - shared across all VAE models"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


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
    def __init__(self, input_dim, cond_in_dim, condition_emb_dim = 10, latent_dim=128, hidden_dim=512, n_classes=3):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_emb_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        
        # Encoder: input modality + condition
        encoder_input_dim = input_dim
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

        self.encoder_condition = nn.Sequential(
            nn.Linear(cond_in_dim, condition_emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(condition_emb_dim),
        )
        
        encoder_output_dim = hidden_dim // 2 + condition_emb_dim
        # Latent space parameters
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
        
        # Decoder: reconstruct input modality
        decoder_input_dim = latent_dim + condition_emb_dim
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
    
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def encode(self, x, condition):
        """Encode inputs to latent space"""
        h = self.encoder(x)
        h_cond = self.encoder_condition(condition)
        h = torch.cat([h, h_cond], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h_cond
    
    def decode(self, z, cond_emb):
        """Decode from latent space"""
        z_cond = torch.cat([z, cond_emb], dim=1)
        x_recon = self.decoder(z_cond)
        return x_recon
    
    def forward(self, x, condition):
        """Forward pass"""
        mu, logvar, cond_emb = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, cond_emb)
        tissue_pred = self.classifier(z)
        return x_recon, mu, logvar, z, tissue_pred


class MultiModalConditionalVAE(BaseConditionalVAE):
    """
    Multi-modal Conditional VAE Model.
    
    Processes both TCR embeddings and GEX data together with conditional information.
    
    Args:
        tcr_dim: Dimension of TCR embeddings
        gex_dim: Dimension of GEX data
        condition_emb_dim: Dimension of condition embedding (default: 10)
        latent_dim: Dimension of latent space (default: 128)
        hidden_dim: Dimension of hidden layers (default: 512)
        n_classes: Number of classification classes (default: 3)
    """
    def __init__(self, tcr_dim, gex_dim, cond_in_dim, condition_emb_dim=10, latent_dim=128, hidden_dim=512, n_classes=3):
        super().__init__()
        
        self.tcr_dim = tcr_dim
        self.gex_dim = gex_dim
        self.condition_dim = condition_emb_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        
        # Encoder: combine both modalities
        self.encoder_tcr = nn.Sequential(
            nn.Linear(tcr_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
        )

        self.encoder_gex = nn.Sequential(
            nn.Linear(gex_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
        )

        self.encoder_condition = nn.Sequential(
            nn.Linear(cond_in_dim, condition_emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(condition_emb_dim),
        )
        
        # Latent space parameters - account for both modalities + condition + additional_features
        encoder_output_dim = hidden_dim // 2 + hidden_dim // 2  + condition_emb_dim
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
        
        # Decoder: reconstruct both modalities
        decoder_input_dim = latent_dim + condition_emb_dim
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
        tcr_h = self.encoder_tcr(tcr)
        gex_h = self.encoder_gex(gex)
        h_cond = self.encoder_condition(condition)
        h = torch.cat([tcr_h, gex_h, h_cond], dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, h_cond
    
    def decode(self, z, cond_emb):
        """Decode from latent space"""
        z_cond = torch.cat([z, cond_emb], dim=1)
        tcr_recon = self.decoder_tcr(z_cond)
        gex_recon = self.decoder_gex(z_cond)
        return tcr_recon, gex_recon
    
    def forward(self, tcr, gex, condition):
        """Forward pass"""
        mu, logvar, cond_emb = self.encode(tcr, gex, condition)
        z = self.reparameterize(mu, logvar)
        tcr_recon, gex_recon = self.decode(z, cond_emb)
        tissue_pred = self.classifier(z)
        return tcr_recon, gex_recon, mu, logvar, z, tissue_pred


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class TransformerMultiModalConditionalVAE(BaseConditionalVAE):
    """
    Multi-modal VAE with transformer encoders for both TCR and GEX data.
    
    Uses self-attention transformer to process both TCR and GEX modalities before VAE encoding.
    TCR uses positional encoding, GEX does not.
    
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

    """
    def __init__(self, tcr_dim, gex_dim, condition_emb_dim=10, latent_dim=128, hidden_dim=512, 
                 n_classes=3, transformer_d_model=512, transformer_nhead=8, 
                 transformer_num_layers=2, transformer_dim_feedforward=1024, 
                 transformer_dropout=0.1):
        super().__init__()
        
        self.tcr_dim = tcr_dim
        self.gex_dim = gex_dim
        self.condition_dim = condition_emb_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        
        # Store transformer config
        self.transformer_d_model = transformer_d_model

        # Projection from tcr_dim to transformer_d_model (needed for dimension matching)
        if tcr_dim != transformer_d_model:
            self.tcr_proj = nn.Linear(tcr_dim, transformer_d_model)
        else:
            self.tcr_proj = nn.Identity()
        
        # Sinusoidal positional encoding (non-trainable)
        self.tcr_pos_encoding = PositionalEncoding(transformer_d_model)

        # Transformer encoder for TCR using nn.TransformerEncoderLayer
        tcr_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True,
        )

        self.tcr_transformer = nn.TransformerEncoder(
            encoder_layer=tcr_encoder_layer,
            num_layers=transformer_num_layers,
        )
        
        # Projection from gex_dim to transformer_d_model (needed for dimension matching)
        if gex_dim != transformer_d_model:
            self.gex_proj = nn.Linear(gex_dim, transformer_d_model)
        else:
            self.gex_proj = nn.Identity()
        
        # Transformer encoder for GEX using nn.TransformerEncoderLayer (no positional encoding)
        gex_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True,
        )

        self.gex_transformer = nn.TransformerEncoder(
            encoder_layer=gex_encoder_layer,
            num_layers=transformer_num_layers,
        )
        
        # VAE Encoder: combine transformer output (TCR) + transformer output (GEX) + condition
        encoder_input_dim = transformer_d_model + transformer_d_model + condition_emb_dim
        
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
        decoder_input_dim = latent_dim + condition_emb_dim
        
        # Projection from decoder input (z + condition) to transformer dimension for TCR decoder
        self.tcr_decoder_proj = nn.Linear(decoder_input_dim, transformer_d_model)
        
        # Transformer decoder for TCR using nn.TransformerDecoderLayer
        tcr_decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True,
        )
        
        self.tcr_decoder = nn.TransformerDecoder(
            decoder_layer=tcr_decoder_layer,
            num_layers=transformer_num_layers,
        )
        
        # Projection from transformer output back to TCR dimension
        self.tcr_decoder_output = nn.Linear(transformer_d_model, tcr_dim)
        
        # Projection from decoder input (z + condition) to transformer dimension for GEX decoder
        self.gex_decoder_proj = nn.Linear(decoder_input_dim, transformer_d_model)
        
        # Transformer decoder for GEX using nn.TransformerDecoderLayer (no positional encoding)
        gex_decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True,
        )
        
        self.gex_decoder = nn.TransformerDecoder(
            decoder_layer=gex_decoder_layer,
            num_layers=transformer_num_layers,
        )
        
        # Projection from transformer output back to GEX dimension
        self.gex_decoder_output = nn.Linear(transformer_d_model, gex_dim)
        
        # Classifier head for tissue prediction
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def encode(self, tcr, gex, condition):
        """Encode inputs to latent space using transformer for both TCR and GEX"""
        # Process TCR through transformer
        # Project to transformer dimension
        tcr_emb = self.tcr_proj(tcr)  # (batch_size, d_model)
        
        # Add a singleton sequence dimension so each sample is treated as a 1-token sequence
        tcr_seq = tcr_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Apply sinusoidal positional encoding (non-trainable)
        tcr_seq = self.tcr_pos_encoding(tcr_seq)  # (batch_size, 1, d_model)
        
        # Process TCR through transformer encoder
        tcr_encoded = self.tcr_transformer(tcr_seq)  # (batch_size, 1, d_model)
        
        # Remove sequence dimension
        tcr_transformed = tcr_encoded[:, 0, :]  # (batch_size, d_model)
        
        # Process GEX through transformer (no positional encoding)
        # Project to transformer dimension
        gex_emb = self.gex_proj(gex)  # (batch_size, d_model)
        
        # Add a singleton sequence dimension so each sample is treated as a 1-token sequence
        gex_seq = gex_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Process GEX through transformer encoder (no positional encoding applied)
        gex_encoded = self.gex_transformer(gex_seq)  # (batch_size, 1, d_model)
        
        # Remove sequence dimension
        gex_transformed = gex_encoded[:, 0, :]  # (batch_size, d_model)
        
        # Combine transformer output (TCR) + transformer output (GEX) + condition
        x = torch.cat([tcr_transformed, gex_transformed, condition], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, tcr_encoded, gex_encoded  # Return encoded TCR and GEX for decoder memory
    
    def decode(self, z, condition, tcr_encoded=None, gex_encoded=None):
        """Decode from latent space using TransformerDecoder for both TCR and GEX"""
        z_cond = torch.cat([z, condition], dim=1)
        
        # Decode TCR
        # Project decoder input to transformer dimension
        tcr_decoder_input = self.tcr_decoder_proj(z_cond)  # (batch_size, transformer_d_model)
        
        # Add sequence dimension
        tcr_decoder_input = tcr_decoder_input.unsqueeze(1)  # (batch_size, 1, transformer_d_model)
        
        # Apply positional encoding to decoder input
        tcr_decoder_input = self.tcr_pos_encoding(tcr_decoder_input)  # (batch_size, 1, transformer_d_model)
        
        # Use encoded TCR as memory if available, otherwise use the decoder input as memory
        if tcr_encoded is not None:
            memory = tcr_encoded  # (batch_size, 1, transformer_d_model)
        else:
            # If no encoded TCR available (e.g., during inference), use decoder input as memory
            memory = tcr_decoder_input
        
        # Apply TransformerDecoder
        tcr_decoded = self.tcr_decoder(tcr_decoder_input, memory)  # (batch_size, 1, transformer_d_model)
        
        # Remove sequence dimension and project back to TCR dimension
        tcr_decoded = tcr_decoded[:, 0, :]  # (batch_size, transformer_d_model)
        tcr_recon = self.tcr_decoder_output(tcr_decoded)  # (batch_size, tcr_dim)
        
        # Decode GEX (no positional encoding)
        # Project decoder input to transformer dimension
        gex_decoder_input = self.gex_decoder_proj(z_cond)  # (batch_size, transformer_d_model)
        
        # Add sequence dimension
        gex_decoder_input = gex_decoder_input.unsqueeze(1)  # (batch_size, 1, transformer_d_model)
        
        # No positional encoding for GEX decoder input
        
        # Use encoded GEX as memory if available, otherwise use the decoder input as memory
        if gex_encoded is not None:
            memory = gex_encoded  # (batch_size, 1, transformer_d_model)
        else:
            # If no encoded GEX available (e.g., during inference), use decoder input as memory
            memory = gex_decoder_input
        
        # Apply TransformerDecoder
        gex_decoded = self.gex_decoder(gex_decoder_input, memory)  # (batch_size, 1, transformer_d_model)
        
        # Remove sequence dimension and project back to GEX dimension
        gex_decoded = gex_decoded[:, 0, :]  # (batch_size, transformer_d_model)
        gex_recon = self.gex_decoder_output(gex_decoded)  # (batch_size, gex_dim)
        
        return tcr_recon, gex_recon
    
    def forward(self, tcr, gex, condition):
        """Forward pass"""
        mu, logvar, tcr_encoded, gex_encoded = self.encode(tcr, gex, condition)
        z = self.reparameterize(mu, logvar)
        tcr_recon, gex_recon = self.decode(z, condition, tcr_encoded, gex_encoded)
        tissue_pred = self.classifier(z)
        return tcr_recon, gex_recon, mu, logvar, z, tissue_pred



