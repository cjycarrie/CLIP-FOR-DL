import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image
from config import DATA_PATH, MODEL_CONFIG, AUGMENTATION_CONFIG, DEVICE
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict

def process_multiple_labels(problems_column):
    """处理多个诊断标签
    
    Args:
        problems_column: 包含诊断信息的列
        
    Returns:
        list: 所有唯一的诊断标签列表
    """
    all_labels = []
    for problems in problems_column:
        if isinstance(problems, str):
            # 使用分号分割多个诊断
            labels = [p.strip() for p in problems.split(';')]
            all_labels.extend(labels)
    return list(set(all_labels))

def get_data_transforms():
    """获取数据转换
    
    Returns:
        train_transform: 训练数据转换
        val_transform: 验证数据转换
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(AUGMENTATION_CONFIG['rotation_degrees']),
        transforms.RandomAffine(
            degrees=0,
            translate=AUGMENTATION_CONFIG['translate']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=AUGMENTATION_CONFIG['normalize_mean'],
            std=AUGMENTATION_CONFIG['normalize_std']
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=AUGMENTATION_CONFIG['normalize_mean'],
            std=AUGMENTATION_CONFIG['normalize_std']
        )
    ])
    
    return train_transform, val_transform

def preprocess_image(image_path, image_size):
    """预处理图像
    
    Args:
        image_path: 图像路径
        image_size: 目标图像大小
        
    Returns:
        处理后的图像
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 调整大小
        img = cv2.resize(img, (image_size, image_size))
        
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return np.zeros((image_size, image_size, 3), dtype=np.uint8)

class ChestXrayDataset(Dataset):
    """胸部X光数据集类"""
    
    def __init__(self, reports_df, projections_df, image_dir, transform=None, image_size=224, all_labels=None):
        # Merge on uid to maintain examination context
        self.data = pd.merge(reports_df, projections_df, on='uid')
        self.image_dir = image_dir
        self.transform = transform
        self.image_size = image_size
        self.all_labels = all_labels
        
        # Group by uid to ensure we process each examination together
        self.uids = self.data['uid'].unique()
        
        print(f"Dataset using {len(self.all_labels)} labels")
    
    def __len__(self):
        return len(self.uids)
    
    def __getitem__(self, idx):
        uid = self.uids[idx]
        
        # Get all views for this examination
        exam_views = self.data[self.data['uid'] == uid]
        
        # Get the report data (should be the same for all views of this uid)
        report_data = exam_views.iloc[0]
        
        images = []
        view_types = []
        
        # First try to get frontal and lateral views
        frontal_view = exam_views[exam_views['projection'].str.contains('frontal', case=False, na=False)]
        lateral_view = exam_views[exam_views['projection'].str.contains('lateral', case=False, na=False)]
        
        # Process frontal view
        if not frontal_view.empty:
            img_path = os.path.join(self.image_dir, frontal_view.iloc[0]['filename'])
            img = self.load_and_preprocess_image(img_path)
            if self.transform:
                img = self.transform(img)
            images.append(img)
            view_types.append('frontal')
        
        # Process lateral view
        if not lateral_view.empty:
            img_path = os.path.join(self.image_dir, lateral_view.iloc[0]['filename'])
            img = self.load_and_preprocess_image(img_path)
            if self.transform:
                img = self.transform(img)
            images.append(img)
            view_types.append('lateral')
        
        # If we don't have both views, duplicate the available view
        if len(images) == 0:
            # If no views available, create a blank image
            blank = torch.zeros(3, self.image_size, self.image_size)
            images = [blank, blank]
            view_types = ['unknown', 'unknown']
        elif len(images) == 1:
            # If only one view, duplicate it
            images.append(images[0])
            view_types.append(view_types[0])
        
        # Stack views together
        images = torch.stack(images)
        
        # Get labels directly from disease columns
        labels = torch.tensor([report_data[col] for col in self.all_labels], dtype=torch.float)
        
        # Get findings text
        findings = report_data['findings'] if pd.notna(report_data['findings']) else ""
        
        return images, labels, findings, view_types
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            return img
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

class MultiViewFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(MODEL_CONFIG['shared_embedding_size'] * 2, MODEL_CONFIG['shared_embedding_size']),  # Concatenate frontal and lateral
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(MODEL_CONFIG['shared_embedding_size'], MODEL_CONFIG['shared_embedding_size'])
        )
    
    def forward(self, frontal_view, lateral_view):
        combined = torch.cat([frontal_view, lateral_view], dim=1)
        return self.fusion(combined)

def prepare_data(test_size=0.2, random_state=42):
    """准备训练和验证数据
    
    Args:
        test_size: 验证集比例
        random_state: 随机种子
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        all_labels: 所有标签列表
        reports_df: 完整的报告数据
    """
    print("读取标注数据：" + os.path.join(DATA_PATH['base_dir'], 'indiana_reports_labeled.csv'))
    reports_df = pd.read_csv(os.path.join(DATA_PATH['base_dir'], 'indiana_reports_labeled.csv'))
    
    print("读取投影数据：" + os.path.join(DATA_PATH['base_dir'], 'indiana_projections.csv'))
    projections_df = pd.read_csv(os.path.join(DATA_PATH['base_dir'], 'indiana_projections.csv'))
    
    # 获取疾病标签列
    disease_columns = ['Cardiomegaly', 'Pulmonary Atelectasis', 'Pleural Effusion', 
                      'Nodule', 'Infiltrate', 'Emphysema', 'Thickening', 'Hernia',
                      'Pulmonary Edema', 'Pneumonia', 'Consolidation', 'Pneumothorax',
                      'Fibrosis', 'Mass', 'Granuloma', 'Normal']
    
    print(f"使用 {len(disease_columns)} 个疾病标签")
    
    # 分割数据
    train_reports, val_reports = train_test_split(
        reports_df, 
        test_size=test_size,
        random_state=random_state
    )
    
    # 获取数据转换
    train_transform, val_transform = get_data_transforms()
    
    # 创建数据集
    train_dataset = ChestXrayDataset(
        train_reports,
        projections_df,
        os.path.join(DATA_PATH['image_dir']),
        transform=train_transform,
        image_size=MODEL_CONFIG['image_size'],
        all_labels=disease_columns
    )
    
    val_dataset = ChestXrayDataset(
        val_reports,
        projections_df,
        os.path.join(DATA_PATH['image_dir']),
        transform=val_transform,
        image_size=MODEL_CONFIG['image_size'],
        all_labels=disease_columns
    )
    
    print(f"训练集大小: {len(train_dataset)} 检查")
    print(f"验证集大小: {len(val_dataset)} 检查")
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, disease_columns, reports_df

def load_data():
    """加载数据集
    
    Returns:
        tuple: (训练数据集, 验证数据集, 标签列表, 报告数据)
    """
    return prepare_data()

def get_enhanced_text_features(disease, view_type, tokenizer, text_model):
    prompts = [
        f"This {view_type} view chest X-ray shows {disease}.",
        f"In the {view_type} projection, there are signs of {disease}.",
        f"The {view_type} chest radiograph demonstrates {disease}."
    ]
    # Process prompts...

def predict_with_dynamic_threshold(image_features, text_features, disease_stats):
    similarities = (image_features @ text_features.T)
    
    # Adjust threshold based on disease frequency
    thresholds = []
    for disease, stats in disease_stats.iterrows():
        if stats['frequency'] > 100:
            thresholds.append(0.4)  # Lower threshold for common diseases
        else:
            thresholds.append(0.6)  # Higher threshold for rare diseases
            
    predictions = similarities > torch.tensor(thresholds).to(similarities.device)
    return predictions

def get_disease_cooccurrence_matrix(reports_df):
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    for _, row in reports_df.iterrows():
        diseases = row['Problems'].split(';')
        for d1 in diseases:
            for d2 in diseases:
                if d1 != d2:
                    cooccurrence[d1][d2] += 1
                    
    return cooccurrence

# Use in prediction
def adjust_predictions(raw_predictions, cooccurrence_matrix):
    adjusted_predictions = raw_predictions.clone()
    for i, pred in enumerate(raw_predictions):
        if pred.sum() == 1:  # Single prediction
            disease = disease_list[pred.argmax()]
            # Check common co-occurrences
            for cooccur_disease, count in cooccurrence_matrix[disease].items():
                if count > threshold:
                    idx = disease_list.index(cooccur_disease)
                    adjusted_predictions[i, idx] = 1
    return adjusted_predictions

if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 准备数据
    prepare_data() 
