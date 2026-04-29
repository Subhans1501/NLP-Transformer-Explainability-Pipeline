# NLP-Transformer-Explainability-Pipeline

## Overview
This repository contains a production-grade Natural Language Processing (NLP) pipeline designed to fine-tune Transformer architectures (BERT) for binary sentiment classification on the massive Amazon Polarity dataset. 

Engineered to address real-world deployment challenges, the codebase incorporates practical MLOps methodologies to handle strict hardware constraints, utilizing gradient checkpointing and mixed-precision (FP16) training. Beyond achieving high baseline classification metrics, this project heavily emphasizes **Explainable AI (XAI)** to ensure enterprise-level trust and transparency. It features custom modules to extract and visualize multi-layer attention matrices, alongside SHAP and LIME integrations to mathematically unpack and validate individual model predictions.

## Repository Structure
* `data/` (Ignored): Target directory for processed Hugging Face dataset splits.
* `notebooks/`: Contains the Kaggle training execution record and the local attention extraction notebook.
* `outputs/`: 
  * `attention_maps/`: Multi-layer token relationship heatmaps (PNG).
  * `explainability/`: Interactive SHAP and LIME text prediction evaluations (HTML).
  * `saved_models/` (Ignored): Target directory for local model weight storage.
* `report/`: The comprehensive technical methodology and analytical whitepaper (PDF).
* `src/`: Modular Python scripts for data preparation, model training, and XAI execution.

## Environment Setup
1. Clone the repository to your local machine.
2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Reproduce Results

**1. Data Preparation**
Run the following command to download, subset, tokenize, and split the dataset for local/compute-constrained environments:

```bash
python src/data_prep.py
```

**2. Model Training**
To fine-tune the BERT model (Note: recommended to run on GPU instances like Kaggle Dual-T4):

```bash
python src/train.py
```

**3. Attention Analysis & Explainability**
* To extract and visualize attention heatmaps locally, execute the cells in `notebooks/02_attention_analysis.ipynb`.
* To generate the interactive SHAP and LIME HTML plots, run:

```bash
python src/explainability.py
```

## Hardware & Software Documentation
* **Training Compute:** Kaggle Dual NVIDIA T4 GPUs (15GB VRAM) leveraging FP16 mixed precision.
* **Inference/XAI Compute:** Local CPU environment.
* **Core Libraries:** PyTorch, Transformers, Datasets, SHAP, LIME, Matplotlib, Seaborn.