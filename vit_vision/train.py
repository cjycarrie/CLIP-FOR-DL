import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
import time
from tqdm import tqdm
import cv2
import numpy as np
import torchvision.models as models
import os
import logging
from config import MODEL_CONFIG, TRAINING_CONFIG, LOG_CONFIG, DEVICE, DATA_PATH, MPS_CONFIG
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, classification_report
from disease_analysis import (
    predict_zero_shot, 
    evaluate_predictions, 
    get_training_text_features,
    get_prediction_text_features,
    get_text_features_with_findings as get_text_features
)
import pandas as pd
from prepare_data import load_data
from typing import Union, List, Tuple, Optional, Dict
from PIL import Image
from sklearn.model_selection import train_test_split
import psutil

class ImageTextDataset(Dataset):
    def __init__(self, image_filenames, labels, image_size=224):

        self.image_filenames = image_filenames
        self.labels = labels
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(DATA_PATH['image_dir'], self.image_filenames[idx])
        image = self.preprocess_image(image_path)
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"无法读取图像: {image_path}")
            # 返回一个空白图像
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

class ImageProjection(nn.Module):
    
    def __init__(self, image_embedding_size, shared_embedding_size):
        super(ImageProjection, self).__init__()
        self.image_projection = nn.Linear(image_embedding_size, shared_embedding_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(shared_embedding_size, shared_embedding_size)
        self.dropout = nn.Dropout(MODEL_CONFIG['dropout_rate'])
        self.layer_norm = nn.LayerNorm(shared_embedding_size)

    def forward(self, image_embeddings):
        # Flatten if needed
        if len(image_embeddings.shape) > 2:
            batch_size = image_embeddings.size(0)
            image_embeddings = image_embeddings.view(batch_size, -1)
        
        projected = self.image_projection(image_embeddings)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TextProjection(nn.Module):
    
    def __init__(self, text_embedding_size, shared_embedding_size):
        super(TextProjection, self).__init__()
        self.text_projection = nn.Linear(text_embedding_size, shared_embedding_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(shared_embedding_size, shared_embedding_size)
        self.dropout = nn.Dropout(MODEL_CONFIG['dropout_rate'])
        self.layer_norm = nn.LayerNorm(shared_embedding_size)

    def forward(self, text_embeddings):
        projected = self.text_projection(text_embeddings)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def contrastive_clip_loss_function(text_projection, image_projection, temperature=MODEL_CONFIG['temperature'], mode="eval"):

    logits = (text_projection @ image_projection.T) / temperature
    if mode == "train":
        images_similarity = image_projection @ image_projection.T
        texts_similarity = text_projection @ text_projection.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()
    elif mode == "eval":
        return logits
    else:
        logging.error("Invalid mode for contrastive loss")
        return None

def contrastive_loss(image_features, text_features, temperature=1.0):
  
    logits = (image_features @ text_features.T) / temperature
    
    batch_size = image_features.size(0)
    labels = torch.arange(batch_size).to(DEVICE)
    
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2.0

def multilabel_contrastive_loss(image_features, text_features, labels, temperature=1.0):
 
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    
    similarities = (image_features @ text_features.T) / temperature
    
    if torch.isnan(similarities).any() or torch.isinf(similarities).any():
        logging.error(f"相似度计算出现NaN或Inf: {similarities}")
    
    batch_size = labels.size(0)
    num_classes = text_features.size(0)
    
    if labels.size(1) != num_classes:
        logging.error(f"标签格式不正确: shape={labels.shape}, 期望shape=[{batch_size}, {num_classes}]")
        new_labels = torch.zeros(batch_size, num_classes, device=labels.device)
        new_labels[:, :labels.size(1)] = labels
        labels = new_labels
    
    similarities_clipped = torch.clamp(similarities, -50.0, 50.0)
    pos_probs = torch.sigmoid(similarities_clipped)
    neg_probs = 1 - pos_probs
    
    pos_loss = -torch.sum(torch.log(pos_probs + 1e-8) * labels) / (torch.sum(labels) + 1e-8)
    neg_loss = -torch.sum(torch.log(neg_probs + 1e-8) * (1 - labels)) / (torch.sum(1 - labels) + 1e-8)
    
    loss = (pos_loss + neg_loss) / 2.0
    
    if torch.isnan(loss) or torch.isinf(loss) or loss > 1000:
        logging.error(f"多标签对比损失异常: {loss}")
        logging.error(f"正样本损失: {pos_loss}, 负样本损失: {neg_loss}")
        return contrastive_loss(image_features, text_features, temperature)
    
    return loss

def calculate_accuracy(predictions, labels):
 
    sample_accuracy = ((predictions == labels).float().mean(dim=1) * 100).mean().item()
    
    label_accuracy = ((predictions == labels).float().mean(dim=0) * 100).mean().item()
    
    return sample_accuracy, label_accuracy

def calculate_multilabel_metrics(predictions, labels):
   
    pred_labels = (predictions > 0.5).float()
    
    sample_accuracy = ((pred_labels == labels).float().mean(dim=1) * 100).mean().item()
    
    label_accuracy = ((pred_labels == labels).float().mean(dim=0) * 100).mean().item()
    
    hamming_score = (pred_labels == labels).float().mean().item() * 100
    
    exact_match = (pred_labels == labels).all(dim=1).float().mean().item() * 100
    
    top1_pred = predictions.argmax(dim=1)
    top1_acc = torch.any(labels[torch.arange(len(labels)), top1_pred] == 1, dim=0).float().mean().item() * 100
    
    _, top3_preds = predictions.topk(k=min(3, predictions.size(1)), dim=1)
    top3_acc = torch.any(labels.gather(1, top3_preds) == 1, dim=1).float().mean().item() * 100
    
    pred_pos = pred_labels.sum(dim=1)
    true_pos = labels.sum(dim=1)
    correct_pos = (pred_labels * labels).sum(dim=1)
    
    precision = correct_pos / (pred_pos + 1e-8)
    recall = correct_pos / (true_pos + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    avg_f1 = f1.mean().item() * 100
    
    return {
        'sample_acc': sample_accuracy,    
        'label_acc': label_accuracy,     
        'hamming_score': hamming_score,   
        'exact_match': exact_match,       
        'top1_acc': top1_acc,            
        'top3_acc': top3_acc,            
        'f1_score': avg_f1               
    }

def train_epoch(models, train_loader, optimizer, epoch, disease_list):

    for name, model in models.items():
        if name != 'tokenizer' and hasattr(model, 'train'):
            model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    end = time.time()
    total_batches = len(train_loader)
    
    logging.info(f'Epoch: [{epoch}] - Starting training...')
    
    with torch.no_grad():
        text_features = get_text_features(
            disease_list,
            models['tokenizer'],
            models['text_model'],
            models['text_projector']
        ).to(DEVICE)
    
    for i, (images, labels, findings, view_types) in enumerate(train_loader):
        try:
            data_time.update(time.time() - end)
            
            if DEVICE.type == 'mps':
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                logging.info(f"Batch {i} - Initial memory usage: {mem_before:.2f} MB")
            
            if DEVICE.type == 'mps':
                try:
                    torch.mps.empty_cache()
                    
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    views = images.size(1)
                    all_view_features = []
                    
                    for v in range(views):
                        view_images = images[:, v]
                        view_embeddings = models['resnet'](view_images)
                        view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                        view_features = models['image_projector'](view_embeddings)
                        all_view_features.append(view_features)
                        
                        mem_after_view = process.memory_info().rss / (1024 * 1024)
                        logging.info(f"Batch {i}, View {v} - Memory usage: {mem_after_view:.2f} MB")
                    
                    image_features = models['view_fusion'](all_view_features[0], all_view_features[1])
                    
                    mem_after_fusion = process.memory_info().rss / (1024 * 1024)
                    logging.info(f"Batch {i} - Memory after fusion: {mem_after_fusion:.2f} MB")
                    
                except Exception as e:
                    logging.warning(f"MPS处理失败，回退到CPU: {str(e)}")
                    images = images.to('cpu')
                    labels = labels.to('cpu')
                    
                    for name, model in models.items():
                        if name != 'tokenizer' and hasattr(model, 'to'):
                            model.to('cpu')
                    
                    views = images.size(1)
                    all_view_features = []
                    
                    for v in range(views):
                        view_images = images[:, v]
                        view_embeddings = models['resnet'](view_images)
                        view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                        view_features = models['image_projector'](view_embeddings)
                        all_view_features.append(view_features)
                    
                    image_features = models['view_fusion'](all_view_features[0], all_view_features[1])
                    
                    image_features = image_features.to(DEVICE)
                    labels = labels.to(DEVICE)
                    for name, model in models.items():
                        if name != 'tokenizer' and hasattr(model, 'to'):
                            model.to(DEVICE)
            else:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                views = images.size(1)
                all_view_features = []
                
                for v in range(views):
                    view_images = images[:, v]
                    view_embeddings = models['resnet'](view_images)
                    view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                    view_features = models['image_projector'](view_embeddings)
                    all_view_features.append(view_features)
                
                image_features = models['view_fusion'](all_view_features[0], all_view_features[1])
            
            loss = multilabel_contrastive_loss(image_features, text_features, labels)
            
            with torch.no_grad():
                similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
                predictions = (torch.sigmoid(similarities) > 0.5).float()
                
                correct_predictions = (predictions == labels).float().mean(dim=1)
                accuracy = correct_predictions.mean().item() * 100
                
                class_accuracies = (predictions == labels).float().mean(dim=0)
                class_acc_dict = {disease: acc.item() * 100 for disease, acc in zip(disease_list, class_accuracies)}
            
            optimizer.zero_grad()
            loss.backward()
            
            if DEVICE.type == 'mps':
                for param in optimizer.param_groups[0]['params']:
                    if param.grad is not None and param.grad.device != DEVICE:
                        param.grad = param.grad.to(DEVICE)
                
                mem_after_backward = process.memory_info().rss / (1024 * 1024)
                logging.info(f"Batch {i} - Memory after backward: {mem_after_backward:.2f} MB")
            
            optimizer.step()
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            if DEVICE.type == 'mps':
                torch.mps.empty_cache()
                mem_end = process.memory_info().rss / (1024 * 1024)
                logging.info(f"Batch {i} - Final memory usage: {mem_end:.2f} MB")
            
            if i % 10 == 0 or i == total_batches - 1:
                logging.info(
                    f'Epoch: [{epoch}][{i}/{total_batches}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Acc {accuracies.val:.2f}% ({accuracies.avg:.2f}%)'
                )
                
                if i % 50 == 0:
                    logging.info("Per-class accuracies:")
                    for disease, acc in class_acc_dict.items():
                        logging.info(f"{disease}: {acc:.2f}%")
        
        except Exception as e:
            logging.error(f"Error in batch {i}: {str(e)}")
            continue
    
    return losses.avg, accuracies.avg

def validate(models, val_loader, disease_list):
  
    for name, model in models.items():
        if name != 'tokenizer' and hasattr(model, 'eval'):
            model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_predictions = []
    all_labels = []
    

    with torch.no_grad():
        text_features = get_text_features(
            disease_list,
            models['tokenizer'],
            models['text_model'],
            models['text_projector']
        )
    
    with torch.no_grad():
        for images, labels, findings, view_types in val_loader:
            if DEVICE.type == 'mps':
                images = images.to('cpu')
                labels = labels.to('cpu')
                
                for name, model in models.items():
                    if name != 'tokenizer' and hasattr(model, 'to'):
                        model.to('cpu')
                
                views = images.size(1)
                all_view_features = []
                
                for v in range(views):
                    view_images = images[:, v]
                    view_embeddings = models['resnet'](view_images)
                    
                    if isinstance(view_embeddings, torch.Tensor):
                        view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                    
                    view_features = models['image_projector'](view_embeddings)
                    all_view_features.append(view_features)
                
                all_view_features = [f.to(DEVICE) for f in all_view_features]
                labels = labels.to(DEVICE)
                
                for name, model in models.items():
                    if name != 'tokenizer' and hasattr(model, 'to'):
                        model.to(DEVICE)
            else:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                views = images.size(1)
                all_view_features = []
                
                for v in range(views):
                    view_images = images[:, v]
                    view_embeddings = models['resnet'](view_images)
                    
                    if isinstance(view_embeddings, torch.Tensor):
                        view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                    
                    view_features = models['image_projector'](view_embeddings)
                    all_view_features.append(view_features)
            
            image_features = models['view_fusion'](all_view_features[0], all_view_features[1])
            
            loss = multilabel_contrastive_loss(image_features, text_features, labels)
            
            similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
            predictions = (torch.sigmoid(similarities) > 0.5).float()
            
            correct_predictions = (predictions == labels).float().mean(dim=1)
            accuracy = correct_predictions.mean().item() * 100
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    class_accuracies = (all_predictions == all_labels).float().mean(dim=0)
    class_acc_dict = {disease: acc.item() * 100 for disease, acc in zip(disease_list, class_accuracies)}
    
    logging.info(f"\nValidation Results:")
    logging.info(f"Average Loss: {losses.avg:.4f}")
    logging.info(f"Average Accuracy: {accuracies.avg:.2f}%")
    logging.info("\nPer-class Accuracies:")
    for disease, acc in class_acc_dict.items():
        logging.info(f"{disease}: {acc:.2f}%")
    
    return losses.avg, accuracies.avg

def save_checkpoint(state, is_best, checkpoint_dir):
 
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    
    epoch_filename = os.path.join(checkpoint_dir, f'model_epoch_{state["epoch"]}.pth')
    torch.save(state, epoch_filename)
    
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filename)

def load_checkpoint(models, optimizer, filename):
  
    if not os.path.exists(filename):
        return 0, float('inf')
    
    try:
        checkpoint = torch.load(filename)
        
        for name, model in models.items():
            if name in checkpoint['models']:
                try:
                    model.load_state_dict(checkpoint['models'][name])
                except RuntimeError as e:
                    logging.warning(f"无法加载模型 {name} 的状态: {str(e)}")
                    logging.warning(f"将使用新初始化的模型继续训练")
                    continue
        
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            logging.warning("无法加载优化器状态，将使用新初始化的优化器")
        
        logging.info(f'成功加载检查点: {filename}')
        return checkpoint['epoch'], checkpoint['loss']
    except Exception as e:
        logging.warning(f"加载检查点时出错: {str(e)}")
        return 0, float('inf')

def predict_and_evaluate(models, val_loader, disease_list):
 
    for model in models.values():
        model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Predicting"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            image_embeddings = models['resnet'](images)
            image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
            image_features = models['image_projector'](image_embeddings)
            
            text_features = models['text_features']
            
            predictions = predict_multilabel(
                image_features,
                text_features,
                threshold=0.5
            )
            
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return predictions, labels

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_clip(train_loader, val_loader, disease_list, reports_df):
   
    models = initialize_models(DEVICE)
    
    enhanced_prompts = create_enhanced_prompts_with_findings(reports_df)
    
    try:
        text_features = get_text_features(
            disease_list,
            models['tokenizer'],
            models['text_model'],
            models['text_projector'],
            enhanced_prompts=enhanced_prompts
        )
        models['text_features'] = text_features
    except Exception as e:
        logging.error(f"计算文本特征时出错: {e}")
        raise
    
    params = []
    for name, model in models.items():
        if name not in ['tokenizer', 'text_features'] and hasattr(model, 'parameters'):
            params.extend(list(model.parameters()))
    
    optimizer = torch.optim.AdamW(
        params,
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    checkpoint_path = os.path.join(LOG_CONFIG['checkpoint_dir'], 'checkpoint.pth')
    start_epoch, best_val_loss = load_checkpoint(models, optimizer, checkpoint_path)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_CONFIG['epochs'],
        eta_min=TRAINING_CONFIG['min_learning_rate'],
        last_epoch=start_epoch-1 if start_epoch > 0 else -1
    )
    
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    patience_counter = 0
    
    for epoch in range(start_epoch, TRAINING_CONFIG['epochs']):
        train_loss, train_acc = train_epoch(models, train_loader, optimizer, epoch, disease_list)
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        
        val_loss, val_acc = validate(models, val_loader, disease_list)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

        scheduler.step()
        
        logging.info(
            f'Epoch {epoch} - '
            f'Train Loss: {train_loss:.4f}, '
            f'Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, '
            f'Val Acc: {val_acc:.2f}%, '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        model_states = {}
        for name, model in models.items():
            if name != 'tokenizer' and name != 'text_features' and hasattr(model, 'state_dict'):
                model_states[name] = model.state_dict()
        
        state = {
            'epoch': epoch + 1,
            'models': model_states,
            'optimizer': optimizer.state_dict(),
            'loss': val_loss,
            'accuracy': val_acc
        }
        
        save_checkpoint(state, is_best, LOG_CONFIG['checkpoint_dir'])
        
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            logging.info(f'早停: {patience_counter} epochs没有改善')
            break
    
    return models, train_history

def predict_multilabel(image_features, text_features, threshold=0.5):
   
    similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
    probabilities = torch.sigmoid(similarities)
    predictions = (probabilities > threshold).float()
    return predictions

def initialize_models(device):
  
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()  
    resnet = resnet.to(device)
    
    image_projector = ImageProjection(
        MODEL_CONFIG['image_embedding_size'],
        MODEL_CONFIG['shared_embedding_size']
    ).to(device)
    
    view_fusion = MultiViewFusion().to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    text_model = AutoModel.from_pretrained(MODEL_CONFIG['model_name']).to(device)
    
    text_projector = TextProjection(
        MODEL_CONFIG['text_embedding_size'],
        MODEL_CONFIG['shared_embedding_size']
    ).to(device)
    
    return {
        'resnet': resnet,
        'image_projector': image_projector,
        'view_fusion': view_fusion,
        'tokenizer': tokenizer,
        'text_model': text_model,
        'text_projector': text_projector
    }

def get_text_features(diseases, tokenizer, text_model, text_projector, enhanced_prompts=None, device=DEVICE):

    if enhanced_prompts:
        prompts = [f"{prompt} {disease}." for disease, prompt in zip(diseases, enhanced_prompts)]
    else:
        prompts = [f"This is a chest X-ray showing {disease}." for disease in diseases]
    
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding='max_length',
        max_length=MODEL_CONFIG['max_text_length'],
        truncation=True
    )
    
    if device.type == 'mps' and MPS_CONFIG['fallback_to_cpu']:
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        text_model = text_model.to('cpu')
        text_projector = text_projector.to('cpu')
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = text_model(**inputs)
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        text_features = text_projector(text_embeddings)
        text_features = F.normalize(text_features, dim=-1)
    
    if device.type == 'mps' and MPS_CONFIG['fallback_to_cpu']:
        text_features = text_features.to(device)
        text_model = text_model.to(device)
        text_projector = text_projector.to(device)
    
    return text_features

def create_enhanced_prompts_with_findings(reports_df):
    
    return ["Enhanced prompt for " + disease for disease in reports_df['Problems'].unique()]

class MultiViewFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(MODEL_CONFIG['shared_embedding_size'] * 2, MODEL_CONFIG['shared_embedding_size']),  # Concatenate frontal and lateral
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(MODEL_CONFIG['shared_embedding_size'], MODEL_CONFIG['shared_embedding_size'])
        )
    
    def forward(self, frontal_view, lateral_view):
        combined = torch.cat([frontal_view, lateral_view], dim=1)
        return self.fusion(combined)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_CONFIG['log_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    os.makedirs(LOG_CONFIG['log_dir'], exist_ok=True)
    os.makedirs(LOG_CONFIG['checkpoint_dir'], exist_ok=True)
    
    logging.info("Loading data...")
    train_loader, val_loader, disease_list, reports_df = load_data()
    
    logging.info("Starting training...")
    models, history = train_clip(train_loader, val_loader, disease_list, reports_df)
    
    logging.info("Saving training history...")
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(LOG_CONFIG['log_dir'], 'training_history.csv'), index=False)
    
    logging.info("Training completed!") 
