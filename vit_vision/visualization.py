
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix
from config import MODEL_CONFIG, LOG_CONFIG
import os

def display_image(image_np, title=None):
   
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    return plt

def save_or_show_image(plt, filename=None):
  
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def plot_training_history(train_losses, val_losses, filename=None):
  
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    save_or_show_image(plt, filename)

def plot_confusion_matrix(y_true, y_pred, classes, filename=None):
  
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    save_or_show_image(plt, filename)

def visualize_top_predictions(image, predictions, scores, filename=None):
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Chest X-ray Image')
    
    y_pos = np.arange(len(predictions))
    ax2.barh(y_pos, scores, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(predictions)
    ax2.invert_yaxis() 
    ax2.set_xlabel('Confidence Score')
    ax2.set_title('Top Predictions')
    
    plt.tight_layout()
    save_or_show_image(plt, filename)

def visualize_disease_distribution(disease_stats, filename=None):
   
    top_diseases = disease_stats.sort_values('frequency', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='frequency', y='disease', data=top_diseases)
    ax.set_title('Top 20 Disease Frequency')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Disease')
    
    for i, v in enumerate(top_diseases['frequency']):
        ax.text(v + 0.1, i, str(int(v)), color='black', va='center')
    
    plt.tight_layout()
    save_or_show_image(plt, filename)

def plot_metrics_comparison(metrics_dict, title='Performance Metrics', filename=None):
   
    numeric_metrics = {k: v for k, v in metrics_dict.items() 
                       if isinstance(v, (int, float)) and k != 'classification_report'}
    
    plt.figure(figsize=(10, 6))
    plt.bar(numeric_metrics.keys(), numeric_metrics.values())
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    for i, (k, v) in enumerate(numeric_metrics.items()):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    save_or_show_image(plt, filename)

def visualize_image_text_similarity(image_features, text_features, disease_list, filename=None):
  
    similarity = (image_features @ text_features.T).detach().cpu().numpy()
    
    if len(similarity.shape) > 1 and similarity.shape[0] > 1:
        similarity = similarity[0]
    
    top_indices = np.argsort(similarity)[::-1][:10]
    top_diseases = [disease_list[i] for i in top_indices]
    top_scores = similarity[top_indices]
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=top_scores, y=top_diseases)
    ax.set_title('Top 10 Similar Diseases')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Disease')
    
    for i, v in enumerate(top_scores):
        ax.text(v + 0.01, i, f'{v:.4f}', color='black', va='center')
    
    plt.tight_layout()
    save_or_show_image(plt, filename)

def visualize_predictions(images, predictions, scores, disease_list, save_dir):
   
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (image, preds, confs) in enumerate(zip(images, predictions, scores)):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image.transpose(1, 2, 0))  # CHW -> HWC
        plt.axis('off')
        plt.title('Input X-ray Image')
        
        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(preds))
        plt.barh(y_pos, confs)
        plt.yticks(y_pos, preds)
        plt.xlabel('Confidence')
        plt.title('Top Predictions')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
        plt.close()

def plot_training_history(train_loss, val_loss, save_path):
   
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, save_path):
   
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metrics_comparison(metrics, title, save_path):
   
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1']
    values = [metrics[m] for m in metrics_to_plot]
    
    plt.bar(metrics_to_plot, values)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_disease_distribution(disease_stats, save_path):
   
    plt.figure(figsize=(15, 8))
    sns.barplot(x='disease', y='frequency', data=disease_stats)
    plt.xticks(rotation=45, ha='right')
    plt.title('Disease Distribution in Dataset')
    plt.xlabel('Disease')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 