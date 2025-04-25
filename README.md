# Chest X-ray Analysis with CLIP

This project implements a CLIP-based model for analyzing chest X-ray images and their associated medical findings. The model is trained to understand the relationship between medical images and their textual descriptions, enabling zero-shot classification of various chest conditions.

## Features

- CLIP-based image-text matching for chest X-ray analysis
- Zero-shot classification of multiple chest conditions
- GradCAM visualization for model interpretability
- Training with validation metrics tracking
- Checkpoint saving and loading functionality

## Requirements

- Python 3.8+
- PyTorch 2.5.1+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/cjycarrie/CLIP-FOR-DL.git
cd CLIP-FOR-DL
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:
```bash
python train.py
```

The training script will:
- Load and preprocess the chest X-ray dataset
- Train the CLIP model on image-text pairs
- Save checkpoints during training
- Log training metrics

### Zero-shot Prediction

To perform zero-shot predictions:
```bash
python zero_shot_predict.py
```

### Visualization

To visualize model predictions and attention:
```bash
python gradcam.py
```

## Model Architecture

The project uses a CLIP-based architecture that consists of:
- Image Encoder: Modified Vision Transformer
- Text Encoder: Transformer-based encoder
- Projection heads for alignment of image and text features

## File Structure

- `train.py`: Main training script
- `config.py`: Configuration parameters
- `gradcam.py`: Implementation of GradCAM visualization
- `zero_shot_predict.py`: Zero-shot prediction script

## Project Structure

```
.
├── config.py              # Configuration file for model parameters and training settings
├── train.py              # Main training script with model definition and training loop
├── gradcam.py           # GradCAM visualization implementation
├── zero_shot_predict.py # Zero-shot prediction implementation
│
├── data/                # Data directory (not included in repository)
│   ├── indiana_reports.csv     # Diagnostic report data
│   ├── indiana_projections.csv # Projection data
│   └── images_normalized/      # Normalized X-ray images
│
├── logs/                # Output directory (not included in repository)
│   └── training.log     # Training logs
│
└── checkpoints/         # Model checkpoints directory (not included in repository)
    ├── checkpoint.pth   # Latest checkpoint
    └── model_best.pth   # Best model checkpoint
```

## Core Files Description

### train.py
- Model architecture implementation
- Training and validation loops
- Loss function implementation
- Checkpoint management

### config.py
- Data path configuration
- Model parameters
- Training parameters
- Logging configuration

### gradcam.py
- GradCAM visualization implementation
- Attention map generation
- Visual explanation tools

### zero_shot_predict.py
- Zero-shot prediction implementation
- Disease classification
- Performance evaluation

