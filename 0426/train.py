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
        """初始化数据集
        
        Args:
            image_filenames: 图像文件名列表
            labels: 标签张量 [num_samples, num_classes]
            image_size: 图像大小
        """
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
    """图像投影模块"""
    
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
    """文本投影模块"""
    
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
    """计算交叉熵损失"""
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def contrastive_clip_loss_function(text_projection, image_projection, temperature=MODEL_CONFIG['temperature'], mode="eval"):
    """计算CLIP对比损失
    
    Args:
        text_projection: 文本投影特征
        image_projection: 图像投影特征
        temperature: 温度参数
        mode: 'train' 或 'eval'
        
    Returns:
        train模式下返回损失值，eval模式下返回相似度矩阵
    """
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
    """计算对比损失
    
    Args:
        image_features: 图像特征 [batch_size, embedding_dim]
        text_features: 文本特征 [num_classes, embedding_dim]
        temperature: 温度参数
        
    Returns:
        损失值
    """
    # 计算相似度矩阵 [batch_size, num_classes]
    logits = (image_features @ text_features.T) / temperature
    
    # 创建标签（对角线为1，其他为0）
    batch_size = image_features.size(0)
    labels = torch.arange(batch_size).to(DEVICE)
    
    # 计算交叉熵损失
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    return (loss_i + loss_t) / 2.0

def multilabel_contrastive_loss(image_features, text_features, labels, temperature=1.0):
    """计算多标签对比损失
    
    Args:
        image_features: 图像特征 [batch_size, embedding_dim]
        text_features: 文本特征 [num_classes, embedding_dim]
        labels: 多标签标签 [batch_size, num_classes]
        temperature: 温度参数
        
    Returns:
        损失值
    """
    # 规范化特征
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    
    # 计算相似度，值域为[-1/temp, 1/temp]
    similarities = (image_features @ text_features.T) / temperature
    
    # 记录similarities的统计信息，帮助调试
    if torch.isnan(similarities).any() or torch.isinf(similarities).any():
        logging.error(f"相似度计算出现NaN或Inf: {similarities}")
    
    batch_size = labels.size(0)
    num_classes = text_features.size(0)
    
    # 确保标签格式正确
    if labels.size(1) != num_classes:
        logging.error(f"标签格式不正确: shape={labels.shape}, 期望shape=[{batch_size}, {num_classes}]")
        # 创建新的标签张量，填充缺失的类别
        new_labels = torch.zeros(batch_size, num_classes, device=labels.device)
        new_labels[:, :labels.size(1)] = labels
        labels = new_labels
    
    # 计算正样本和负样本的损失，使用clip防止数值溢出
    similarities_clipped = torch.clamp(similarities, -50.0, 50.0)
    pos_probs = torch.sigmoid(similarities_clipped)
    neg_probs = 1 - pos_probs
    
    # 计算损失
    pos_loss = -torch.sum(torch.log(pos_probs + 1e-8) * labels) / (torch.sum(labels) + 1e-8)
    neg_loss = -torch.sum(torch.log(neg_probs + 1e-8) * (1 - labels)) / (torch.sum(1 - labels) + 1e-8)
    
    loss = (pos_loss + neg_loss) / 2.0
    
    # 检查损失值是否合理
    if torch.isnan(loss) or torch.isinf(loss) or loss > 1000:
        logging.error(f"多标签对比损失异常: {loss}")
        logging.error(f"正样本损失: {pos_loss}, 负样本损失: {neg_loss}")
        # 回退到基本对比损失
        return contrastive_loss(image_features, text_features, temperature)
    
    return loss

def calculate_accuracy(predictions, labels):
    """计算多标签分类的准确率指标
    
    Args:
        predictions: 预测结果 [batch_size, num_classes]
        labels: 真实标签 [batch_size, num_classes]
    
    Returns:
        sample_accuracy: 样本级别的准确率（每个样本预测正确的标签比例的平均值）
        label_accuracy: 标签级别的准确率（所有标签的准确率平均值）
    """
    # 样本级别的准确率：每个样本预测正确的标签比例的平均值
    sample_accuracy = ((predictions == labels).float().mean(dim=1) * 100).mean().item()
    
    # 标签级别的准确率：所有标签的准确率平均值
    label_accuracy = ((predictions == labels).float().mean(dim=0) * 100).mean().item()
    
    return sample_accuracy, label_accuracy

def calculate_multilabel_metrics(predictions, labels):
    """计算多标签分类的多个评估指标
    
    Args:
        predictions: 预测结果 [batch_size, num_classes]
        labels: 真实标签 [batch_size, num_classes]
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 将预测概率转换为二值预测
    pred_labels = (predictions > 0.5).float()
    
    # 1. Sample Accuracy: 每个样本预测正确的标签比例的平均值
    sample_accuracy = ((pred_labels == labels).float().mean(dim=1) * 100).mean().item()
    
    # 2. Label Accuracy: 每个标签的预测准确率的平均值
    label_accuracy = ((pred_labels == labels).float().mean(dim=0) * 100).mean().item()
    
    # 3. Hamming Score: 预测标签与真实标签的匹配程度
    hamming_score = (pred_labels == labels).float().mean().item() * 100
    
    # 4. Exact Match Ratio: 完全匹配的样本比例
    exact_match = (pred_labels == labels).all(dim=1).float().mean().item() * 100
    
    # 5. Top-1 Accuracy: 最高置信度的预测是否在真实标签中
    top1_pred = predictions.argmax(dim=1)
    top1_acc = torch.any(labels[torch.arange(len(labels)), top1_pred] == 1, dim=0).float().mean().item() * 100
    
    # 6. Top-3 Accuracy: 预测的前3个标签中是否包含真实标签
    _, top3_preds = predictions.topk(k=min(3, predictions.size(1)), dim=1)
    top3_acc = torch.any(labels.gather(1, top3_preds) == 1, dim=1).float().mean().item() * 100
    
    # 7. 样本级别F1分数
    pred_pos = pred_labels.sum(dim=1)
    true_pos = labels.sum(dim=1)
    correct_pos = (pred_labels * labels).sum(dim=1)
    
    precision = correct_pos / (pred_pos + 1e-8)
    recall = correct_pos / (true_pos + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    avg_f1 = f1.mean().item() * 100
    
    return {
        'sample_acc': sample_accuracy,    # 样本级准确率
        'label_acc': label_accuracy,      # 标签级准确率
        'hamming_score': hamming_score,   # 汉明得分
        'exact_match': exact_match,       # 完全匹配率
        'top1_acc': top1_acc,            # Top-1准确率
        'top3_acc': top3_acc,            # Top-3准确率
        'f1_score': avg_f1               # F1分数
    }

def train_epoch(models, train_loader, optimizer, epoch, disease_list):
    """训练一个epoch
    
    Args:
        models: 模型字典
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前epoch
        disease_list: 疾病列表
        
    Returns:
        float: 平均损失值
    """
    # 设置训练模式
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
    
    # 预计算文本特征
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
            
            # 记录每个批次开始时的内存使用
            if DEVICE.type == 'mps':
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / (1024 * 1024)  # MB
                logging.info(f"Batch {i} - Initial memory usage: {mem_before:.2f} MB")
            
            # 将数据移到设备上
            if DEVICE.type == 'mps':
                try:
                    # 清除可能的缓存
                    torch.mps.empty_cache()
                    
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    # 处理每个视角
                    views = images.size(1)
                    all_view_features = []
                    
                    for v in range(views):
                        view_images = images[:, v]
                        # 确保视图数据在正确的设备上
                        view_embeddings = models['resnet'](view_images)
                        view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                        view_features = models['image_projector'](view_embeddings)
                        all_view_features.append(view_features)
                        
                        # 记录每个视角处理后的内存使用
                        mem_after_view = process.memory_info().rss / (1024 * 1024)
                        logging.info(f"Batch {i}, View {v} - Memory usage: {mem_after_view:.2f} MB")
                    
                    # 融合特征
                    image_features = models['view_fusion'](all_view_features[0], all_view_features[1])
                    
                    # 记录特征融合后的内存使用
                    mem_after_fusion = process.memory_info().rss / (1024 * 1024)
                    logging.info(f"Batch {i} - Memory after fusion: {mem_after_fusion:.2f} MB")
                    
                except Exception as e:
                    logging.warning(f"MPS处理失败，回退到CPU: {str(e)}")
                    # 如果MPS处理失败，回退到CPU
                    images = images.to('cpu')
                    labels = labels.to('cpu')
                    
                    # 将模型临时移到CPU
                    for name, model in models.items():
                        if name != 'tokenizer' and hasattr(model, 'to'):
                            model.to('cpu')
                    
                    # 在CPU上处理
                    views = images.size(1)
                    all_view_features = []
                    
                    for v in range(views):
                        view_images = images[:, v]
                        view_embeddings = models['resnet'](view_images)
                        view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                        view_features = models['image_projector'](view_embeddings)
                        all_view_features.append(view_features)
                    
                    # 融合特征
                    image_features = models['view_fusion'](all_view_features[0], all_view_features[1])
                    
                    # 将结果和模型移回MPS
                    image_features = image_features.to(DEVICE)
                    labels = labels.to(DEVICE)
                    for name, model in models.items():
                        if name != 'tokenizer' and hasattr(model, 'to'):
                            model.to(DEVICE)
            else:
                # 对于其他设备，直接使用
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # 处理每个视角
                views = images.size(1)
                all_view_features = []
                
                for v in range(views):
                    view_images = images[:, v]
                    view_embeddings = models['resnet'](view_images)
                    view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                    view_features = models['image_projector'](view_embeddings)
                    all_view_features.append(view_features)
                
                # 融合特征
                image_features = models['view_fusion'](all_view_features[0], all_view_features[1])
            
            # 计算损失
            loss = multilabel_contrastive_loss(image_features, text_features, labels)
            
            # 计算预测结果
            with torch.no_grad():
                similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
                predictions = (torch.sigmoid(similarities) > 0.5).float()
                
                # 计算准确率（样本级别的准确率）
                correct_predictions = (predictions == labels).float().mean(dim=1)
                accuracy = correct_predictions.mean().item() * 100
                
                # 计算每个类别的准确率
                class_accuracies = (predictions == labels).float().mean(dim=0)
                class_acc_dict = {disease: acc.item() * 100 for disease, acc in zip(disease_list, class_accuracies)}
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 如果在MPS上，确保梯度在正确的设备上
            if DEVICE.type == 'mps':
                for param in optimizer.param_groups[0]['params']:
                    if param.grad is not None and param.grad.device != DEVICE:
                        param.grad = param.grad.to(DEVICE)
                
                # 记录反向传播后的内存使用
                mem_after_backward = process.memory_info().rss / (1024 * 1024)
                logging.info(f"Batch {i} - Memory after backward: {mem_after_backward:.2f} MB")
            
            optimizer.step()
            
            # 更新统计信息
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 如果使用MPS，在每个批次结束时清理缓存
            if DEVICE.type == 'mps':
                torch.mps.empty_cache()
                mem_end = process.memory_info().rss / (1024 * 1024)
                logging.info(f"Batch {i} - Final memory usage: {mem_end:.2f} MB")
            
            # 打印进度
            if i % 10 == 0 or i == total_batches - 1:
                logging.info(
                    f'Epoch: [{epoch}][{i}/{total_batches}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Acc {accuracies.val:.2f}% ({accuracies.avg:.2f}%)'
                )
                
                # 每50个批次打印一次详细的类别准确率
                if i % 50 == 0:
                    logging.info("Per-class accuracies:")
                    for disease, acc in class_acc_dict.items():
                        logging.info(f"{disease}: {acc:.2f}%")
        
        except Exception as e:
            logging.error(f"Error in batch {i}: {str(e)}")
            continue
    
    return losses.avg, accuracies.avg

def validate(models, val_loader, disease_list):
    """验证模型
    
    Args:
        models: 模型字典
        val_loader: 验证数据加载器
        disease_list: 疾病列表
        
    Returns:
        验证损失和准确率
    """
    # 设置评估模式
    for name, model in models.items():
        if name != 'tokenizer' and hasattr(model, 'eval'):
            model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_predictions = []
    all_labels = []
    
    # 预计算文本特征
    with torch.no_grad():
        text_features = get_text_features(
            disease_list,
            models['tokenizer'],
            models['text_model'],
            models['text_projector']
        )
    
    with torch.no_grad():
        for images, labels, findings, view_types in val_loader:
            # 将数据移到设备上
            if DEVICE.type == 'mps':
                # 对于MPS设备，先在CPU上准备数据
                images = images.to('cpu')
                labels = labels.to('cpu')
                
                # 确保模型在正确的设备上
                for name, model in models.items():
                    if name != 'tokenizer' and hasattr(model, 'to'):
                        model.to('cpu')
                
                # 处理每个视角
                views = images.size(1)
                all_view_features = []
                
                for v in range(views):
                    view_images = images[:, v]
                    view_embeddings = models['resnet'](view_images)
                    
                    if isinstance(view_embeddings, torch.Tensor):
                        view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                    
                    view_features = models['image_projector'](view_embeddings)
                    all_view_features.append(view_features)
                
                # 将处理后的特征移到MPS设备
                all_view_features = [f.to(DEVICE) for f in all_view_features]
                labels = labels.to(DEVICE)
                
                # 将模型移回MPS设备
                for name, model in models.items():
                    if name != 'tokenizer' and hasattr(model, 'to'):
                        model.to(DEVICE)
            else:
                # 对于其他设备，直接使用
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                # 处理每个视角
                views = images.size(1)
                all_view_features = []
                
                for v in range(views):
                    view_images = images[:, v]
                    view_embeddings = models['resnet'](view_images)
                    
                    if isinstance(view_embeddings, torch.Tensor):
                        view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                    
                    view_features = models['image_projector'](view_embeddings)
                    all_view_features.append(view_features)
            
            # 融合特征
            image_features = models['view_fusion'](all_view_features[0], all_view_features[1])
            
            # 计算损失
            loss = multilabel_contrastive_loss(image_features, text_features, labels)
            
            # 计算预测结果
            similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
            predictions = (torch.sigmoid(similarities) > 0.5).float()
            
            # 计算准确率
            correct_predictions = (predictions == labels).float().mean(dim=1)
            accuracy = correct_predictions.mean().item() * 100
            
            # 保存预测结果用于计算整体指标
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
    
    # 计算整体指标
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算每个类别的准确率
    class_accuracies = (all_predictions == all_labels).float().mean(dim=0)
    class_acc_dict = {disease: acc.item() * 100 for disease, acc in zip(disease_list, class_accuracies)}
    
    # 打印验证结果
    logging.info(f"\nValidation Results:")
    logging.info(f"Average Loss: {losses.avg:.4f}")
    logging.info(f"Average Accuracy: {accuracies.avg:.2f}%")
    logging.info("\nPer-class Accuracies:")
    for disease, acc in class_acc_dict.items():
        logging.info(f"{disease}: {acc:.2f}%")
    
    return losses.avg, accuracies.avg

def save_checkpoint(state, is_best, checkpoint_dir):
    """保存模型检查点
    
    Args:
        state: 模型状态字典
        is_best: 是否是最佳模型
        checkpoint_dir: 检查点保存目录
    """
    # 保存当前checkpoint
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, filename)
    
    # 保存每个epoch的模型
    epoch_filename = os.path.join(checkpoint_dir, f'model_epoch_{state["epoch"]}.pth')
    torch.save(state, epoch_filename)
    
    # 如果是最佳模型，保存为model_best.pth
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filename)

def load_checkpoint(models, optimizer, filename):
    """加载检查点
    
    Args:
        models: 模型字典
        optimizer: 优化器
        filename: 加载文件名
        
    Returns:
        开始epoch和最佳损失
    """
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
    """进行预测并评估
    
    Args:
        models: 模型字典
        val_loader: 验证数据加载器
        disease_list: 疾病列表
        
    Returns:
        predictions: 预测结果
        labels: 真实标签
    """
    # 设置评估模式
    for model in models.values():
        model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Predicting"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 获取图像特征
            image_embeddings = models['resnet'](images)
            image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
            image_features = models['image_projector'](image_embeddings)
            
            # 获取文本特征
            text_features = models['text_features']
            
            # 计算预测结果
            predictions = predict_multilabel(
                image_features,
                text_features,
                threshold=0.5
            )
            
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    # 合并所有批次的结果
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return predictions, labels

class AverageMeter:
    """跟踪指标的平均值和当前值"""
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
    """训练CLIP模型
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        disease_list: 疾病列表
        reports_df: 包含Problems和Findings列的DataFrame
        
    Returns:
        models: 训练好的模型
        train_history: 训练历史
    """
    # 初始化模型
    models = initialize_models(DEVICE)
    
    # 创建增强的提示
    enhanced_prompts = create_enhanced_prompts_with_findings(reports_df)
    
    # 预计算文本特征
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
    
    # 设置优化器
    params = []
    for name, model in models.items():
        if name not in ['tokenizer', 'text_features'] and hasattr(model, 'parameters'):
            params.extend(list(model.parameters()))
    
    optimizer = torch.optim.AdamW(
        params,
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # 尝试加载检查点
    checkpoint_path = os.path.join(LOG_CONFIG['checkpoint_dir'], 'checkpoint.pth')
    start_epoch, best_val_loss = load_checkpoint(models, optimizer, checkpoint_path)
    
    # 设置学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TRAINING_CONFIG['epochs'],
        eta_min=TRAINING_CONFIG['min_learning_rate'],
        last_epoch=start_epoch-1 if start_epoch > 0 else -1
    )
    
    # 训练历史
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 早停
    patience_counter = 0
    
    # 训练循环
    for epoch in range(start_epoch, TRAINING_CONFIG['epochs']):
        # 训练一个epoch
        train_loss, train_acc = train_epoch(models, train_loader, optimizer, epoch, disease_list)
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        
        # 验证
        val_loss, val_acc = validate(models, val_loader, disease_list)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
        # 更新学习率
        scheduler.step()
        
        # 记录训练信息
        logging.info(
            f'Epoch {epoch} - '
            f'Train Loss: {train_loss:.4f}, '
            f'Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, '
            f'Val Acc: {val_acc:.2f}%, '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 只为可序列化的模型创建state_dict
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
        
        # 早停检查
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            logging.info(f'早停: {patience_counter} epochs没有改善')
            break
    
    return models, train_history

def predict_multilabel(image_features, text_features, threshold=0.5):
    """多标签预测
    
    Args:
        image_features: 图像特征
        text_features: 文本特征
        threshold: 预测阈值
        
    Returns:
        torch.Tensor: 预测结果
    """
    # 计算相似度
    similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
    # 使用sigmoid将相似度转换为概率
    probabilities = torch.sigmoid(similarities)
    # 根据阈值进行预测
    predictions = (probabilities > threshold).float()
    return predictions

def initialize_models(device):
    """初始化模型
    
    Args:
        device: 计算设备
        
    Returns:
        models: 模型字典
    """
    # 初始化ResNet
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()  # 移除最后的全连接层
    resnet = resnet.to(device)
    
    # 初始化图像投影层
    image_projector = ImageProjection(
        MODEL_CONFIG['image_embedding_size'],
        MODEL_CONFIG['shared_embedding_size']
    ).to(device)
    
    # 初始化视图融合模块
    view_fusion = MultiViewFusion().to(device)
    
    # 初始化文本模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    text_model = AutoModel.from_pretrained(MODEL_CONFIG['model_name']).to(device)
    
    # 初始化文本投影层
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
    """获取疾病文本的特征表示
    
    Args:
        diseases: 疾病列表
        tokenizer: 分词器
        text_model: 文本编码器
        text_projector: 文本投影层
        enhanced_prompts: 增强的提示（可选）
        device: 计算设备
        
    Returns:
        torch.Tensor: 文本特征
    """
    if enhanced_prompts:
        prompts = [f"{prompt} {disease}." for disease, prompt in zip(diseases, enhanced_prompts)]
    else:
        prompts = [f"This is a chest X-ray showing {disease}." for disease in diseases]
    
    # 对文本进行编码
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding='max_length',
        max_length=MODEL_CONFIG['max_text_length'],
        truncation=True
    )
    
    # 如果使用MPS设备，某些操作可能需要在CPU上进行
    if device.type == 'mps' and MPS_CONFIG['fallback_to_cpu']:
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        text_model = text_model.to('cpu')
        text_projector = text_projector.to('cpu')
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取文本特征
    with torch.no_grad():
        outputs = text_model(**inputs)
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        text_features = text_projector(text_embeddings)
        text_features = F.normalize(text_features, dim=-1)
    
    # 如果之前转移到了CPU，现在转回设备
    if device.type == 'mps' and MPS_CONFIG['fallback_to_cpu']:
        text_features = text_features.to(device)
        text_model = text_model.to(device)
        text_projector = text_projector.to(device)
    
    return text_features

def create_enhanced_prompts_with_findings(reports_df):
    """创建增强的提示"""
    # 实现创建增强提示的逻辑
    # 这里需要根据实际的报告数据来实现
    # 这里只是一个占位符，实际实现需要根据数据结构和需求来实现
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
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_CONFIG['log_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # 创建必要的目录
    os.makedirs(LOG_CONFIG['log_dir'], exist_ok=True)
    os.makedirs(LOG_CONFIG['checkpoint_dir'], exist_ok=True)
    
    # 加载数据
    logging.info("Loading data...")
    train_loader, val_loader, disease_list, reports_df = load_data()
    
    # 训练模型
    logging.info("Starting training...")
    models, history = train_clip(train_loader, val_loader, disease_list, reports_df)
    
    # 保存训练历史
    logging.info("Saving training history...")
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(LOG_CONFIG['log_dir'], 'training_history.csv'), index=False)
    
    logging.info("Training completed!") 
