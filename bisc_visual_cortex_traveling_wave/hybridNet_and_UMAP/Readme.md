WaveCNN Transformer Pipeline & Dimensionality Reduction Analysis
================================================================

Overview
--------
This repository focuses on two key aspects:

1. **Hybrid-Decoder Model (CNN-Transformer Pipeline)**
   - Implements a **CNN-Transformer hybrid model** for classifying wave data.
   - Uses CNN for spatial feature extraction and a Transformer for sequence processing.
   - Trains and evaluates the model with provided datasets.

2. **Dimensionality Reduction & Resolution Effect Analysis**
   - Explores the impact of **resolution on neural data representation**.
   - Utilizes **dimensionality reduction techniques** to analyze the structure of wave data.
   - Provides insights into the relationship between resolution and classification accuracy.

Installation
------------
Prerequisites:
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

To install dependencies, run:
```bash
pip install torch numpy matplotlib jupyter
```

File Structure
--------------

### **1. Hybrid-Decoder Model (CNN-Transformer Pipeline)**
Jupyter Notebooks:
- **CNNTransformer_Pipline.ipynb** – Experiments and visualization of CNN-Transformer model results.

Python Scripts:
- **HybridModel.py** – Defines the CNN-Transformer hybrid model with CNN-based feature extraction and a Transformer-based sequence processor.
- **run_pipline.py** – Main script to run the pipeline, including training and evaluation.
- **wave_pipline.py** – Implements the WaveCNNTransformerPipeline class, managing data loading, model training, and evaluation.

### **2. Dimensionality Reduction & Resolution Effect Analysis**
Jupyter Notebooks:
- **DimensionalityReduction_and_ResolutionEffect.ipynb** – Analyzes dimensionality reduction and resolution effects.

Usage
-----

### **1. Running the CNN-Transformer Pipeline**
To train and evaluate the model, run:
```bash
python run_pipline.py
```
This script will:
1. Load the dataset.
2. Train the CNN-Transformer model.
3. Evaluate the model on test data.

#### Training Parameters:
- Modify `run_pipline.py` or `wave_pipline.py` to adjust parameters like epochs and batch size.

#### Model Overview:
- **CNN** extracts spatial features from wave data.
- **Transformer** processes sequential dependencies.
- **HybridModel.py** integrates both into a **Wave Transformer Classifier**.

### **2. Dimensionality Reduction & Resolution Analysis**
To run the dimensionality reduction analysis, open the Jupyter Notebook:
```bash
jupyter notebook DimensionalityReduction_and_ResolutionEffect.ipynb
```
Steps:
1. Load the dataset.
2. Run the provided cells to compute and visualize dimensionality reduction and resolution effects.

Analysis Details
----------------

### **1. Dimensionality Reduction & Resolution Effect Analysis**
- This analysis explores how reducing the dimensionality of wave data affects interpretability and classification performance.
- Techniques used: **PCA, UMAP, and t-SNE**.
- Investigates how different resolutions affect feature separation and clustering in low-dimensional space.
- Results help in understanding whether higher resolution improves classification or if lower-dimensional representations capture key information.

### **2. Why CNN-Transformer?**
- **CNN extracts spatial structure** from wave data, capturing local patterns.
- **Transformer models sequential dependencies**, allowing for better temporal feature extraction.
- The hybrid approach combines both benefits, making it well-suited for wave classification.
- Analysis in `CNNTransformer_Pipline.ipynb` compares performance with baseline models.

### **3. Performance Analysis**
- Evaluates model performance using **Accuracy, Loss, and AUC-ROC curves**.
- Examines how resolution affects classification accuracy.
- Compares CNN-Transformer performance to baseline classifiers (e.g., CNN-only, Transformer-only).
- Findings suggest that **resolution plays a key role** in classification reliability.

### **4. Key Findings from Resolution Analysis**
- Higher resolution generally improves classification accuracy but may lead to overfitting.
- Lower resolution preserves key features but reduces fine-grained detail.
- Dimensionality reduction techniques (PCA, UMAP) reveal distinct feature clustering at different resolutions.
- Model performance varies depending on the chosen resolution level, suggesting an **optimal range** for best results.

### **5. How to Interpret Analysis Results**
- **CNN-Transformer training loss** should decrease over epochs, with validation accuracy stabilizing.
- **Dimensionality reduction plots**: Clusters indicate well-separated features, while overlapping suggests lower discriminability.
- **Resolution effect graphs** show how classification accuracy varies across different input resolutions.
- **t-SNE/UMAP visualizations** should reveal meaningful structure in lower-dimensional space.

Configuration
-------------
Modify key settings:
- `wave_pipline.py` – Adjust model architecture.
- `run_pipline.py` – Change dataset loading.

Example Output
--------------
- The pipeline prints training progress and evaluation metrics.
- `CNNTransformer_Pipline.ipynb` and `DimensionalityReduction_and_ResolutionEffect.ipynb` contain visual plots.
