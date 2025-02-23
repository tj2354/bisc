# run_pipeline.py
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

from wave_pipeline import WaveCNNTransformerPipeline
from utils import load_your_data

# Load Data
wavetrain, wavetest, train_indices, test_indices = load_your_data()
# Run the Pipeline
WaveCNNTransformerPipeline(wavetrain, wavetest, train_indices, test_indices).run(num_epochs=30)
