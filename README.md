# EmotionsMechInt: Mechanistic Interpretability of Emotional Transformer Circuits

## Overview

This repository is dedicated to the mechanistic interpretability (MI) of transformer models, specifically focusing on how they process and represent emotions in text.

By using the TransformerLens library, we aim to reverse-engineer the "circuits"—the specific sequence of neurons, attention heads, and MLP layers—that activate to handle emotional concepts, sentiment, and tonal shifts during the model's forward pass. The goal is to move beyond mere performance metrics to achieve a deep, algorithmic understanding of the model's behavior.

## Prerequisites & Installation

Prerequisites

Conda or Miniconda (Highly Recommended for managing deep learning dependencies)

Python 3.10+

Installation

1. Clone the Repository

git clone [https://github.com/center-for-humans-and-machines/internship-hpc-repo-template](https://github.com/center-for-humans-and-machines/internship-hpc-repo-template) EmotionsMechInt
cd EmotionsMechInt


2. Create and Activate the Conda Environment

This step is crucial for PyTorch and TransformerLens stability.

conda create --name emotions-mech-int python=3.10
conda activate emotions-mech-int


3. Install Dependencies

Install the required libraries from requirements.txt:

pip install -r requirements.txt


⚠️ Important GPU Installation Note (CUDA Users)

The requirements.txt file installs the CPU-only version of PyTorch. If you are running this research on a machine with an NVIDIA GPU (CUDA), you MUST replace the standard installation by following the specific commands on the official PyTorch website for your CUDA version. This step is necessary to ensure optimal performance.

## Project Architecture

The core analysis is modularized into several key components that facilitate the end-to-end interpretability workflow.

Module

Location

Description

EmotionAnalysisPipeline

src/emotional_mi_pipeline.py

The main orchestrator that loads the model, fetches the data, and runs the initial forward-pass hooks necessary to capture internal activations.

Logit Lens Analysis

src/analysis/logit_lens_analysis.py

Implements the Logit Lens technique to inspect the predicted logits at every layer, giving an early view into the model's intermediate predictions.

Causal Validation

src/analysis/causal_validation_analysis.py

Conducts causal interventions (e.g., path patching) to validate the function of hypothesized circuits by measuring the effect of disabling or replacing them.

Attention Weights

src/analysis/attention_weights_analysis.py

Focuses on attention head behavior, visualizing attention patterns and analyzing which tokens are attended to during emotional processing.

Visualisations

src/analysis/mi_visualisations.py

Handles all plotting and data visualization (e.g., heatmaps, scatter plots) for the interpretability findings using Matplotlib/Seaborn.

## Analysis Workflow

The primary entry point for running and exploring the research is the main Jupyter Notebook.

The Main Notebook

All research findings and detailed analysis steps are documented and executed within:

notebooks/Emotion_Circuit_Discovery.ipynb

Example Implementation

The notebook follows the standard MI methodology:

1. Pipeline Setup & Run
```python
from src.emotional_mi_pipeline import EmotionAnalysisPipeline
from src.analysis import LogitLensAnalysis, CausalValidationAnalysis, MiVisualisations```


# Initialize and run the core pipeline
pipeline = EmotionAnalysisPipeline(
    model_name='llama-chat', 
    data_path='data/emotion_prompts.csv'
)

This collects all necessary activations from the model's forward pass
activations = pipeline.run()


2. Intermediate Analysis

The collected activations are passed sequentially to the analysis modules for deep dives:

## 1. Intermediate outputs check
LogitLensAnalysis.run(activations)

## 2. Validate a hypothesized circuit
CausalValidationAnalysis.run(
    model=pipeline.model, 
    circuit_path=[(6, 'attn'), (10, 'mlp'), (12, 'attn')] # Example hypothesized path
)

## 3. Visualize findings
MiVisualisations.plot_head_outputs(activations)
