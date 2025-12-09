"""
Transformer Encoders with Self-Attention for feature processing.

This module provides transformer encoder classes that can be used to process
feature vectors using self-attention mechanisms before feeding them to downstream models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with self-attention for processing feature sequences.
    Processes input features as a sequence and applies self-attention.
    
    Args:
        input_dim: Dimension of input features
        d_model: Dimension of transformer embeddings (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer encoder layers (default: 2)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=2, dim_feedforward=1024, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project input features to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, input_dim) - input feature vector
            
        Returns:
            output: (batch_size, d_model) - transformed features
        """
        batch_size = x.size(0)
        
        # Reshape to sequence: treat entire vector as single token
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch_size, 1, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        
        # Global pooling (mean pooling over sequence dimension)
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Output projection
        output = self.output_projection(x)  # (batch_size, d_model)
        
        return output


class FeatureTransformerEncoder(nn.Module):
    """
    Alternative transformer encoder that processes features as a sequence.
    Chunks the feature vector into segments and treats each segment as a token.
    This is better for long feature sequences.
    
    Args:
        input_dim: Dimension of input features
        d_model: Dimension of transformer embeddings (default: 256)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer encoder layers (default: 2)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout rate (default: 0.1)
        chunk_size: Size of each chunk for FeatureTransformerEncoder (default: 64)
    """
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=2, 
                 dim_feedforward=1024, dropout=0.1, chunk_size=64):
        super(FeatureTransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        # Calculate number of chunks
        self.num_chunks = (input_dim + chunk_size - 1) // chunk_size
        
        # Project each chunk to model dimension
        self.chunk_projection = nn.Linear(chunk_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_chunks, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, input_dim) - input feature vector
            
        Returns:
            output: (batch_size, d_model) - transformed features
        """
        batch_size = x.size(0)
        
        # Pad input to be divisible by chunk_size
        pad_size = (self.num_chunks * self.chunk_size) - x.size(1)
        if pad_size > 0:
            x = F.pad(x, (0, pad_size))
        
        # Reshape into chunks: (batch_size, num_chunks, chunk_size)
        x = x.view(batch_size, self.num_chunks, self.chunk_size)
        
        # Project each chunk
        x = self.chunk_projection(x)  # (batch_size, num_chunks, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, num_chunks, d_model)
        
        # Global pooling (mean pooling)
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Output projection
        output = self.output_projection(x)  # (batch_size, d_model)
        
        return output








