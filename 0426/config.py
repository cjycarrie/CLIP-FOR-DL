"""
Configuration file for CLIP chest X-ray project
"""

import torch
import os

# Data paths
DATA_PATH = {
    'base_dir': 'data',
    'image_dir': 'data/images_normalized',
    'reports_csv': 'indiana_reports.csv',
    'projections_csv': 'indiana_projections.csv',
    'train_data': 'train_data.csv',
    'val_data': 'val_data.csv'
}

# Model parameters
MODEL_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'image_size': 224,
    'num_labels': 16,  # 更新为实际的标签数量：16个疾病类别
    'max_text_length': 512,
    'temperature': 0.07,
    'dropout_rate': 0.1,
    'image_embedding_size': 2048,  # ResNet50的输出维度
    'text_embedding_size': 768,   # Bio_ClinicalBERT的输出维度
    'shared_embedding_size': 512,
    'num_attention_heads': 8,
    'num_transformer_layers': 6,
    'warmup_steps': 1000,
    'weight_decay': 0.01,
    'clip_grad_norm': 1.0,
    'model_name': 'emilyalsentzer/Bio_ClinicalBERT'  # 添加模型名称
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,  # Changed from 1 to 100 for full training
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-6,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'warmup_steps': 1000,
    'validation_interval': 1,
    'early_stopping_patience': 5,
    'scheduler_factor': 0.1,
    'scheduler_patience': 2,
    'num_workers': 4
}

# Device configuration
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# 添加MPS相关配置
MPS_CONFIG = {
    'enabled': DEVICE.type == 'mps',
    'fallback_to_cpu': True  # 当MPS操作不支持时回退到CPU
}

# Logging configuration
LOG_CONFIG = {
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints',
    'log_interval': 100,
    'save_top_k': 3
}

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_degrees': 10,
    'translate': (0.1, 0.1),
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    'random_horizontal_flip_p': 0.5,
    'random_rotation_degrees': 10,
    'random_affine_translate': (0.1, 0.1)
}

# Prediction parameters
PREDICTION_CONFIG = {
    'threshold': 0.5,
    'top_k': 3,
    'min_confidence': 0.3
}

# Create necessary directories
for directory in [DATA_PATH['base_dir'], DATA_PATH['image_dir'], 
                 LOG_CONFIG['log_dir'], LOG_CONFIG['checkpoint_dir']]:
    os.makedirs(directory, exist_ok=True) 
