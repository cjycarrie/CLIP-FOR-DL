import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL_CONFIG

class MultiModalAttention(nn.Module):
    def __init__(self, 
                 image_embedding_size=MODEL_CONFIG['image_embedding_size'],
                 text_embedding_size=MODEL_CONFIG['text_embedding_size'],
                 shared_embedding_size=MODEL_CONFIG['shared_embedding_size'],
                 num_heads=MODEL_CONFIG['num_attention_heads'],
                 dropout=MODEL_CONFIG['dropout_rate']):
        super().__init__()
        
        # 投影层，将图像和文本特征映射到共享空间
        self.image_projection = nn.Linear(image_embedding_size, shared_embedding_size)
        self.text_projection = nn.Linear(text_embedding_size, shared_embedding_size)
        
        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=shared_embedding_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(shared_embedding_size)
        self.layer_norm2 = nn.LayerNorm(shared_embedding_size)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(shared_embedding_size, shared_embedding_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_embedding_size * 4, shared_embedding_size),
            nn.Dropout(dropout)
        )
        
        # 输出投影
        self.output_projection = nn.Linear(shared_embedding_size * 2, shared_embedding_size)
        
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: [batch_size, image_embedding_size]
            text_features: [num_classes, text_embedding_size]
            
        Returns:
            enhanced_features: [batch_size, shared_embedding_size]
            attn_weights: [batch_size, 1, num_classes]
        """
        # 1. 投影到共享空间
        image_proj = self.image_projection(image_features)  # [batch_size, shared_embedding_size]
        text_proj = self.text_projection(text_features)     # [num_classes, shared_embedding_size]
        
        # 2. 准备注意力输入
        # 扩展文本特征以匹配批次大小
        text_proj_expanded = text_proj.unsqueeze(0).expand(image_proj.size(0), -1, -1)
        image_proj_expanded = image_proj.unsqueeze(1)  # [batch_size, 1, shared_embedding_size]
        
        # 3. 跨模态注意力
        # 图像作为query，文本作为key和value
        attn_output, attn_weights = self.multihead_attn(
            query=image_proj_expanded,
            key=text_proj_expanded,
            value=text_proj_expanded
        )
        
        # 4. 残差连接和层归一化
        attn_output = self.layer_norm1(image_proj_expanded + attn_output)
        
        # 5. 前馈网络
        ff_output = self.feed_forward(attn_output.squeeze(1))
        ff_output = self.layer_norm2(attn_output.squeeze(1) + ff_output)
        
        # 6. 特征融合
        # 将原始图像特征和增强后的特征拼接
        enhanced_features = torch.cat([image_features, ff_output], dim=1)
        enhanced_features = self.output_projection(enhanced_features)
        
        return enhanced_features, attn_weights 