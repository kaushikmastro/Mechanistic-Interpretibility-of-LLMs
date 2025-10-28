# EmotionsMechInt: Mechanistic Interpretability of Emotional Transformer Circuits

# Overview

This repository is dedicated to the mechanistic interpretability (MI) of transformer models, specifically focusing on how they process and represent emotions in text.

By using TransformerLens, we aim to reverse-engineer the "circuits"—the specific sequence of neurons, attention heads, and MLP layers—that activate to handle emotional concepts, sentiment, and tonal shifts during the model's forward pass. The goal is to move beyond mere performance metrics to achieve a deep, algorithmic understanding of the model's behavior.


# Prerequisites

Conda or Miniconda (Highly Recommended for managing deep learning dependencies)

Python 3.10+


# Installation

1. Clone the repository:

git clone <URL here>
cd EmotionsMechInt


2. Create and Activate the Conda Environment (Recommended):

This ensures a stable environment, which is crucial for PyTorch and TransformerLens.

conda create --name emotions-mech-int python=3.10
conda activate emotions-mech-int


3. Install Dependencies:

Install the required libraries from requirements.txt:

pip install -r requirements.txt


Important GPU Installation Note (CUDA Users)

The requirements.txt file installs the CPU-only version of PyTorch.

If you are running this research on a machine with an NVIDIA GPU (CUDA), you MUST replace the standard installation by following the specific commands on the official PyTorch website for your CUDA version. This step is necessary to ensure optimal performance.

# Project Architecture

The core analysis is modularized into several key components that facilitate the end-to-end interpretability workflow:

Module

Description

emotional_mi_pipeline.py

The main orchestrator that loads the model, fetches the data, and runs the initial forward-pass hooks necessary for analysis.

logit_lens_analysis.py

Implements the Logit Lens technique to inspect the predicted logits at every layer, giving an early view into the model's intermediate predictions.

causal_validation_analysis.py

Conducts causal interventions (e.g., path patching) to validate the function of hypothesized circuits by measuring the effect of disabling or replacing them.

attention_weights_analysis.py

Focuses on attention head behavior, visualizing attention patterns and analyzing which tokens are attended to during emotional processing.

mi_visualisations.py

Handles all plotting and data visualization (e.g., heatmaps, scatter plots) for the interpretability findings using Matplotlib/Seaborn.

Analysis Workflow

The primary entry point for running and exploring the research is the main Jupyter Notebook.

The Main Notebook

All research findings and analysis steps are documented and executed within:

notebooks/Emotion_Circuit_Discovery.ipynb

This notebook follows the standard MI methodology:

Setup: Initialize the environment, load the model, and define the emotion-specific data.

Pipeline: Instantiate and run the core pipeline:

from src.emotional_mi_pipeline import EmotionAnalysisPipeline
# other imports for analysis modules

pipeline = EmotionAnalysisPipeline(model_name='gpt2-small', data_path='data/emotion_prompts.csv')
activations = pipeline.run()


Analysis: The activations are then passed sequentially to the analysis modules for deep dives:

# 1. Intermediate outputs
LogitLensAnalysis.run(activations)

# 2. Validate a hypothesized circuit
CausalValidationAnalysis.run(model=pipeline.model, circuit_path=...)

# 3. Visualize findings
MiVisualisations.plot_head_outputs(activations)
