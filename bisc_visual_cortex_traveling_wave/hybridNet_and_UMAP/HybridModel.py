# CNNTransformerModelClass.py
"""
--------------------------------------------------------------------------------
Project: WaveCNN Transformer Pipeline
Developer: Erfan Zabeh
Email: erfanzabeh1@gmail.com
Date: February 2025
Description: 
    This script is part of a modular pipeline for a CNN-Transformer model 
    to classify wave data. It includes model definitions, a training pipeline, 
    and a runnable entry point.

--------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import math

# CNN Embedding
class CNNEmbedding(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNNEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, output_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Flatten to [batch_size, output_dim]

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        encoding = self.encoding.to(x.device)
        return x + encoding[:, :x.size(1), :]

# Wave Transformer Classifier
class WaveTransformerClassifier(nn.Module):
    def __init__(self, d_model, nhead, num_layers, hidden_dim, num_classes, dropout=0.1, num_time_points=1000):
        super(WaveTransformerClassifier, self).__init__()
        self.cnn_embedding = CNNEmbedding(input_channels=2, output_dim=d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=num_time_points)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)  # Classification head

    def forward(self, x):
        batch_size, frames, electrodes, channels = x.size()
        x = x.view(batch_size * frames, channels, electrodes)  # Merge batch & frames
        x = self.cnn_embedding(x)
        x = x.view(batch_size, frames, -1)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        class_logits = self.classifier(x)  # [batch_size, frames, num_classes]
        return class_logits
