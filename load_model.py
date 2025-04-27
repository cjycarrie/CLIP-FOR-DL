import torch
from config import MODEL_CONFIG, DEVICE
from train import initialize_models
import torch.nn.functional as F

def get_text_features(disease_list, tokenizer, text_model, text_projector):
    """获取文本特征
    
    Args:
        disease_list: 疾病名称列表
        tokenizer: CLIP tokenizer
        text_model: CLIP text model
        text_projector: 文本投影层
        
    Returns:
        text_features: 文本特征张量
    """
    # 构建提示模板
    prompts = [f"a chest x-ray of {disease.lower()}" for disease in disease_list]
    
    # Tokenize
    text_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    text_tokens = {k: v.to(DEVICE) for k, v in text_tokens.items()}
    
    # 获取文本特征
    text_embeddings = text_model(**text_tokens)
    text_embeddings = text_embeddings.last_hidden_state[:, 0, :]
    text_features = text_projector(text_embeddings)
    
    # 标准化特征
    text_features = F.normalize(text_features, dim=-1)
    
    return text_features

def load_trained_model(checkpoint_path='checkpoints/model_best.pth'):
    """加载训练好的模型
    
    Args:
        checkpoint_path: 模型检查点路径
        
    Returns:
        models: 加载了权重的模型字典
    """
    # 初始化模型架构
    models = initialize_models(DEVICE)
    
    # 加载检查点
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # 加载模型权重
        for name, model in models.items():
            if name in checkpoint['models'] and hasattr(model, 'load_state_dict'):
                model.load_state_dict(checkpoint['models'][name])
                print(f"Successfully loaded {name} model")
        
        print(f"\nModel checkpoint loaded from {checkpoint_path}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Validation loss: {checkpoint['loss']:.4f}")
        print(f"Validation accuracy: {checkpoint['accuracy']:.2f}%")
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None
    
    # 设置为评估模式
    for name, model in models.items():
        if name != 'tokenizer' and hasattr(model, 'eval'):
            model.eval()
    
    return models

def get_model_predictions(models, images, disease_list):
    """使用加载的模型进行预测
    
    Args:
        models: 模型字典
        images: 输入图像张量 [batch_size, views, channels, height, width]
        disease_list: 疾病列表
        
    Returns:
        predictions: 预测结果
        probabilities: 预测概率
    """
    with torch.no_grad():
        try:
            # 获取当前设备
            device = next(models['resnet'].parameters()).device
            
            # 将图像移到正确的设备上
            images = images.to(device)
            
            # 获取图像特征
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
            
            # 获取文本特征
            text_features = get_text_features(
                disease_list,
                models['tokenizer'],
                models['text_model'],
                models['text_projector']
            )
            
            # 确保文本特征在正确的设备上
            if text_features.device != device:
                text_features = text_features.to(device)
            
            # 标准化特征
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # 计算相似度和预测结果
            similarities = image_features @ text_features.T
            probabilities = torch.sigmoid(similarities * 4.0)  # 缩放相似度以获得更好的概率分布
            predictions = (probabilities > 0.5).float()
            
            # 将结果移回CPU
            return predictions.cpu(), probabilities.cpu()
            
        except Exception as e:
            print(f"Error in get_model_predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

if __name__ == "__main__":
    # 示例：如何加载和使用模型
    models = load_trained_model()
    if models is not None:
        print("\nModel loaded successfully and ready for inference!")
        print("\nExample usage:")
        print("1. Load your image data")
        print("2. Call get_model_predictions(models, images, disease_list)")
        print("3. Process the predictions as needed") 