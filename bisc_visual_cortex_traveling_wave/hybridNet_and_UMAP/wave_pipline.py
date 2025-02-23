# wave_pipeline.py
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

License: MIT License
--------------------------------------------------------------------------------
"""

import torch
from torch.utils.data import DataLoader
from dataset import WaveDataset
from Repository.HybridModel import WaveTransformerClassifier
from utils import train_model, plot_metrics, load_your_data

class WaveCNNTransformerPipeline:
    def __init__(self, wavetrain, wavetest, train_indices, test_indices):
        # Set device
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")

        # Create DataLoaders
        self.train_loader = DataLoader(WaveDataset(wavetrain, train_indices), batch_size=32, shuffle=True)
        self.test_loader = DataLoader(WaveDataset(wavetest, test_indices), batch_size=32, shuffle=False)

        # Model Initialization
        self.model = WaveTransformerClassifier(
            d_model=256,
            nhead=4,
            num_layers=4,
            hidden_dim=512,
            num_classes=144,
            dropout=0.1
        ).to(self.device)

        # Optimizer and Scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.00001, weight_decay=10)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def run(self, num_epochs=30):
        # Train Model
        train_losses, test_losses, train_accuracies, test_accuracies = train_model(
            self.model, 
            self.train_loader, 
            self.test_loader, 
            self.optimizer, 
            self.scheduler, 
            num_epochs
        )

        # Plot Metrics
        plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

        # Save Model
        torch.save(self.model.state_dict(), "wave_transformer_classifier.pth")
        print("Model saved successfully.")
