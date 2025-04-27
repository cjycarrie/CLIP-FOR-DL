import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Optional
import re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, f1_score
from config import MODEL_CONFIG, DEVICE
from tqdm import tqdm

def analyze_disease_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    分析疾病在数据集中的分布情况
    
    Args:
        df: 包含Problems列的DataFrame
        
    Returns:
        DataFrame包含每个疾病的统计信息
    """
    # 初始化统计字典
    stats = {}
    
    # 统计每个疾病的出现频率
    for idx, row in df.iterrows():
        problems = row.get('Problems', '')
        if not isinstance(problems, str) or pd.isna(problems):
            continue
            
        diseases = [d.strip() for d in problems.split(';')]
        main_disease = diseases[0] if diseases else None
        
        for idx, disease in enumerate(diseases):
            if disease not in stats:
                stats[disease] = {
                    'total_count': 0,
                    'first_position_count': 0
                }
            
            stats[disease]['total_count'] += 1
            if idx == 0:  # 如果是第一个位置
                stats[disease]['first_position_count'] += 1
    
    # 转换为DataFrame
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    
    # 计算百分比
    stats_df['percentage_as_first'] = (stats_df['first_position_count'] / stats_df['total_count']) * 100
    
    # 添加频率列
    stats_df['frequency'] = stats_df['total_count']
    
    return stats_df

def create_rich_prompts(disease_stats):
    """为每个疾病创建丰富的提示模板
    
    Args:
        disease_stats: 疾病统计信息DataFrame
        
    Returns:
        dict: 疾病到提示模板的映射
    """
    prompts = {}
    for disease, stats in disease_stats.iterrows():
        templates = []
        
        # 基础模板
        templates.extend([
            f"This chest X-ray shows {disease}.",
            f"The radiological findings indicate {disease}.",
            f"The image reveals characteristics of {disease}.",
            f"Diagnostic features of {disease} are present.",
            f"The X-ray demonstrates {disease}."
        ])
        
        # 根据疾病频率添加特定模板
        if stats['frequency'] > 10:  # 高频疾病
            templates.extend([
                f"This is a typical case of {disease}.",
                f"Clear radiological signs of {disease} are visible.",
                f"The X-ray shows classic features of {disease}."
            ])
        elif stats['frequency'] > 5:  # 中频疾病
            templates.extend([
                f"This X-ray exhibits features consistent with {disease}.",
                f"Radiological patterns suggest {disease}."
            ])
        else:  # 低频疾病
            templates.extend([
                f"This X-ray shows possible signs of {disease}.",
                f"Some features in this X-ray may indicate {disease}."
            ])
        
        # 根据是否为主要诊断添加特定模板
        if stats['percentage_as_first'] > 80:  # 主要诊断
            templates.extend([
                f"The primary finding in this chest X-ray is {disease}.",
                f"This X-ray primarily shows {disease}."
            ])
        elif stats['percentage_as_first'] > 50:  # 常见主要诊断
            templates.extend([
                f"One of the main findings in this X-ray is {disease}.",
                f"This X-ray shows significant evidence of {disease}."
            ])
        else:  # 次要诊断
            templates.extend([
                f"Among other findings, this X-ray shows {disease}.",
                f"This X-ray reveals {disease} as one of multiple conditions."
            ])
        
        prompts[disease] = templates
    
    return prompts

def get_training_text_features(findings: str,
                           tokenizer,
                           text_model,
                           text_projector,
                           device=DEVICE) -> torch.Tensor:
    """获取训练时的文本特征（使用Findings）
    
    Args:
        findings: 放射科发现描述
        tokenizer: 分词器
        text_model: 文本编码器
        text_projector: 文本投影层
        device: 计算设备
        
    Returns:
        torch.Tensor: 文本特征
    """
    # 对文本进行编码
    inputs = tokenizer(
        findings,
        return_tensors='pt',
        padding='max_length',
        max_length=MODEL_CONFIG['max_text_length'],
        truncation=True
    ).to(device)
    
    # 获取文本特征
    with torch.no_grad():
        outputs = text_model(**inputs)
        text_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_features = text_projector(text_embeddings)
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features

def get_prediction_text_features(diseases: List[str],
                               tokenizer,
                               text_model,
                               text_projector,
                               device=DEVICE) -> torch.Tensor:
    """获取预测时的文本特征
    
    Args:
        diseases: 标准化的疾病列表，应该是以下之一：
                 Cardiomegaly, Pulmonary Atelectasis, Pleural Effusion,
                 Nodule, Infiltrate, Emphysema, Thickening, Hernia,
                 Pulmonary Edema, Pneumonia, Consolidation, Pneumothorax,
                 Fibrosis, Mass, Granuloma, Normal
        tokenizer: 分词器
        text_model: 文本编码器
        text_projector: 文本投影层
        device: 计算设备
        
    Returns:
        torch.Tensor: 文本特征
    """
    # 为每个疾病创建标准化的提示
    prompts = []
    for disease in diseases:
        if disease == "Normal":
            prompt = "This is a normal chest X-ray without any significant findings."
        else:
            prompt = f"This chest X-ray shows {disease}."
        prompts.append(prompt)
    
    # 对文本进行编码
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding='max_length',
        max_length=MODEL_CONFIG['max_text_length'],
        truncation=True
    ).to(device)
    
    # 获取文本特征
    with torch.no_grad():
        outputs = text_model(**inputs)
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        text_features = text_projector(text_embeddings)
        text_features = F.normalize(text_features, dim=-1)
    
    return text_features

def get_enhanced_text_features(disease: str,
                             prompts: Dict[str, List[str]],
                             tokenizer,
                             text_model,
                             text_projector,
                             device: str) -> torch.Tensor:
    """
    获取增强的文本特征
    
    Args:
        disease: 疾病名称
        prompts: 提示模板字典
        tokenizer: 分词器
        text_model: 文本模型
        text_projector: 文本投影模型
        device: 计算设备
        
    Returns:
        文本特征向量
    """
    # 获取该疾病的所有提示
    disease_prompts = prompts.get(disease, [f"This is a chest X-ray showing {disease}."])
    
    # 编码所有提示
    encoded = tokenizer(
        disease_prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MODEL_CONFIG['max_text_length']
    ).to(device)
    
    # 获取文本特征
    with torch.no_grad():
        outputs = text_model(**encoded)
        text_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_proj = text_projector(text_emb)
        text_proj = F.normalize(text_proj, dim=-1)
    
    # 平均所有提示的特征
    return text_proj.mean(dim=0, keepdim=True)

def predict_multilabel(image_features, text_features, threshold=0.5):
    """多标签预测
    
    Args:
        image_features: 图像特征
        text_features: 文本特征
        threshold: 预测阈值
        
    Returns:
        torch.Tensor: 预测结果，对应16个标准化疾病标签
    """
    # 计算相似度
    similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
    # 使用sigmoid将相似度转换为概率
    probabilities = torch.sigmoid(similarities)
    # 根据阈值进行预测
    predictions = (probabilities > threshold).float()
    return predictions

def get_disease_cooccurrence(df):
    """分析疾病共现关系
    
    Args:
        df: 包含Problems列的DataFrame
        
    Returns:
        pd.DataFrame: 疾病共现矩阵
    """
    # 获取所有唯一疾病
    all_diseases = set()
    for problems in df['Problems'].dropna():
        diseases = [d.strip() for d in problems.split(';')]
        all_diseases.update(diseases)
    
    # 创建共现矩阵
    cooccurrence = pd.DataFrame(0, 
                              index=list(all_diseases),
                              columns=list(all_diseases))
    
    # 统计共现次数
    for problems in df['Problems'].dropna():
        diseases = [d.strip() for d in problems.split(';')]
        for d1 in diseases:
            for d2 in diseases:
                if d1 != d2:
                    cooccurrence.loc[d1, d2] += 1
    
    return cooccurrence

def predict_zero_shot(
    images: Union[torch.Tensor, List[torch.Tensor]], 
    models: Dict, 
    disease_list: List[str], 
    top_k: int = 3,
    prompts: Optional[Dict[str, List[str]]] = None,
    use_enhanced_prompts: bool = False
) -> Union[Tuple[List, List], List[Dict]]:
    """零样本预测疾病
    
    Args:
        images: 图像张量或列表
        models: 模型字典
        disease_list: 标准化疾病列表
        top_k: 返回前k个预测
        prompts: 提示模板字典（可选）
        use_enhanced_prompts: 是否使用增强提示
        
    Returns:
        如果输入是批量图像，返回预测列表和分数列表
        如果输入是单个图像，返回包含疾病名称和置信度的字典列表
    """
    # 设置为评估模式
    for name, model in models.items():
        if hasattr(model, 'eval'):
            model.eval()
    
    with torch.no_grad():
        # 处理图像
        is_batch = isinstance(images, torch.Tensor) and images.dim() == 4
        if not is_batch:
            if isinstance(images, list):
                images = images[0]
            images = images.unsqueeze(0).to(DEVICE)
        elif not images.is_cuda and DEVICE == 'cuda':
            images = images.to(DEVICE)
        
        # 获取图像特征
        image_embeddings = models['resnet'](images)
        image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
        image_features = models['image_projector'](image_embeddings)
        image_features = F.normalize(image_features, dim=-1)
        
        # 获取文本特征
        text_features = get_prediction_text_features(
            disease_list,
            models['tokenizer'],
            models['text_model'],
            models['text_projector']
        )
        
        # 计算相似度并获取预测
        similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
        probabilities = F.softmax(similarities, dim=-1)
        
        # 获取top-k预测
        if is_batch:
            batch_predictions = []
            batch_scores = []
            for i in range(probabilities.size(0)):
                values, indices = probabilities[i].topk(min(top_k, len(disease_list)))
                predictions = [disease_list[idx] for idx in indices.cpu().numpy()]
                scores = values.cpu().numpy()
                batch_predictions.append(predictions)
                batch_scores.append(scores)
            return batch_predictions, batch_scores
        else:
            values, indices = probabilities[0].topk(min(top_k, len(disease_list)))
            predictions = [disease_list[idx] for idx in indices.cpu().numpy()]
            scores = values.cpu().numpy()
            return [
                {"disease": disease, "confidence": float(score)} 
                for disease, score in zip(predictions, scores)
            ]

def evaluate_predictions(predictions, true_labels, disease_list):
    """评估预测结果
    
    Args:
        predictions: 预测结果列表，每个元素为一个疾病列表
        true_labels: 真实标签
        disease_list: 疾病列表
        
    Returns:
        dict: 评估指标
    """
    # 将真实标签转换为多标签格式
    if isinstance(true_labels, pd.DataFrame):
        # 如果是DataFrame，假设labels列已经是正确格式
        y_true = np.array(true_labels['labels'].tolist())
    else:
        # 否则假设是列表或数组
        y_true = np.array(true_labels)
    
    # 将预测结果转换为多标签格式
    y_pred = np.zeros((len(predictions), len(disease_list)))
    for i, preds in enumerate(predictions):
        for pred in preds:
            if pred in disease_list:
                y_pred[i, disease_list.index(pred)] = 1
    
    # 计算各种指标
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # 生成分类报告
    report = classification_report(
        y_true, 
        y_pred,
        target_names=disease_list,
        output_dict=True,
        zero_division=0
    )
    
    metrics['classification_report'] = report
    return metrics

def create_enhanced_prompts_with_findings(df: pd.DataFrame) -> Dict[str, List[str]]:
    """为每个疾病创建包含findings的增强提示模板
    
    Args:
        df: 包含Problems和Findings列的DataFrame
        
    Returns:
        dict: 疾病到提示模板的映射
    """
    prompts = {}
    
    # 对每个样本进行处理
    for _, row in df.iterrows():
        if pd.isna(row['Problems']) or pd.isna(row['Findings']):
            continue
            
        diseases = [d.strip() for d in row['Problems'].split(';')]
        findings = row['Findings'].strip()
        
        # 为每个疾病创建提示
        for disease in diseases:
            if disease not in prompts:
                prompts[disease] = []
            
            # 基础模板
            templates = [
                f"This chest X-ray shows {disease}.",
                f"The radiological findings indicate {disease}, specifically: {findings}",
                f"Based on the following observations: {findings}, this X-ray demonstrates {disease}.",
                f"The X-ray reveals {disease}, characterized by: {findings}",
                f"Diagnostic features seen in this X-ray include: {findings}, indicating {disease}."
            ]
            
            prompts[disease].extend(templates)
    
    return prompts

def get_text_features_with_findings(
    diseases: List[str],
    tokenizer,
    text_model,
    text_projector,
    prompts: Dict[str, List[str]],
    device=DEVICE
) -> torch.Tensor:
    """获取结合findings的文本特征
    
    Args:
        diseases: 疾病列表
        tokenizer: 分词器
        text_model: 文本编码器
        text_projector: 文本投影层
        prompts: 提示模板字典
        device: 计算设备
        
    Returns:
        torch.Tensor: 文本特征
    """
    all_features = []
    
    for disease in diseases:
        # 获取该疾病的所有提示
        disease_prompts = prompts.get(disease, [f"This is a chest X-ray showing {disease}."])
        
        # 对文本进行编码
        inputs = tokenizer(
            disease_prompts,
            return_tensors='pt',
            padding='max_length',
            max_length=MODEL_CONFIG['max_text_length'],
            truncation=True
        ).to(device)
        
        # 获取文本特征
        with torch.no_grad():
            outputs = text_model(**inputs)
            text_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            text_features = text_projector(text_embeddings)
            text_features = F.normalize(text_features, dim=-1)
            
            # 平均所有提示的特征
            avg_feature = text_features.mean(dim=0, keepdim=True)
            all_features.append(avg_feature)
    
    # 拼接所有疾病的特征
    return torch.cat(all_features, dim=0) 