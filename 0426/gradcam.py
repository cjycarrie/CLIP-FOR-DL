import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from config import MODEL_CONFIG, DATA_PATH, LOG_CONFIG
from prepare_data import load_data
from disease_analysis import get_text_features_with_findings as get_text_features, create_rich_prompts, analyze_disease_distribution
from train import initialize_models

class GradCAM:
    def __init__(self, model, target_layer):
        """初始化GradCAM
        
        Args:
            model: CLIP视觉模型
            target_layer: 目标层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        def save_activation(module, input, output):
            self.activations = output.detach()
        
        # 注册前向和后向钩子
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)
    
    def generate_cam(self, input_tensor, target_category=None):
        """生成CAM
        
        Args:
            input_tensor: 输入图像张量
            target_category: 目标类别索引
            
        Returns:
            cam: 类激活图
        """
        # 前向传播
        model_output = self.model(input_tensor)
        
        if target_category is None:
            target_category = torch.argmax(model_output)
        
        # 清除之前的梯度
        self.model.zero_grad()
        
        # 反向传播
        if isinstance(model_output, tuple):
            model_output = model_output[0]  # 取第一个输出
        
        if len(model_output.shape) == 1:
            model_output = model_output.unsqueeze(0)
            
        target = torch.zeros_like(model_output)
        target[0, target_category] = 1
        
        model_output.backward(gradient=target)
        
        # 确保梯度和激活值都存在
        if self.gradients is None or self.activations is None:
            raise ValueError("No gradients or activations found")
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        
        # 生成cam
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]
        
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.cpu().numpy()

def preprocess_image(image_path):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor, image

def show_cam_on_image(img, mask, title, save_path=None):
    """在图像上显示CAM"""
    # 将图像数据转换回0-1范围
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img = np.clip(img, 0, 1)
    
    # 调整热力图大小以匹配原始图像
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f'Original Image\n{title}')
    plt.axis('off')
    
    # 热力图
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Attention Map')
    plt.axis('off')
    
    # 叠加图
    plt.subplot(1, 3, 3)
    plt.imshow(cam)
    plt.title('GradCAM Result')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def main(args):
    # 加载数据
    train_loader, val_loader, disease_list, reports_df = load_data()
    
    # 计算验证集大小
    val_size = len(val_loader.dataset)
    print(f"\nValidation set size: {val_size}")
    
    # 检查样本索引是否有效
    if args.sample_index >= val_size:
        print(f"Error: Sample index {args.sample_index} is out of range. Maximum index is {val_size - 1}")
        return
    
    # 初始化模型
    models = initialize_models('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载训练好的模型
    checkpoint_path = os.path.join(LOG_CONFIG['checkpoint_dir'], 'model_best.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        for name, model in models.items():
            if name in checkpoint['models']:
                model.load_state_dict(checkpoint['models'][name])
                print(f"Loaded {name} from checkpoint")
    
    # 获取样本
    for i, (images, labels, findings, view_types) in enumerate(val_loader):
        if i == args.sample_index:
            # 获取两个视图
            frontal_image = images[0, 0]  # [3, 224, 224]
            lateral_image = images[0, 1]  # [3, 224, 224]
            sample_label = labels[0]
            sample_finding = findings[0]
            
            # 打印所有疾病标签
            print("\nDisease labels for this sample:")
            has_disease = False
            for idx, (disease, label) in enumerate(zip(disease_list, sample_label)):
                if label == 1:
                    print(f"- {disease}")
                    has_disease = True
            if not has_disease:
                print("No diseases found (Normal case)")
            print()
            break
    
    # 获取对应的疾病
    positive_diseases = [disease_list[i] for i, label in enumerate(sample_label) if label == 1]
    if not positive_diseases:
        positive_diseases = ['Normal']
    
    # 为每个阳性疾病生成GradCAM
    resnet_model = models['resnet']
    grad_cam = GradCAM(resnet_model, resnet_model.layer4[-1])
    
    # 确保输出目录存在
    os.makedirs('logs/gradcam', exist_ok=True)
    
    # 创建提示模板
    disease_stats = analyze_disease_distribution(reports_df)
    prompts = create_rich_prompts(disease_stats)
    
    for disease in positive_diseases:
        print(f"\nGenerating GradCAM for disease: {disease}")
        print(f"Related findings: {sample_finding}")
        
        # 获取疾病的文本特征
        text_features = get_text_features(
            [disease],
            models['tokenizer'],
            models['text_model'],
            models['text_projector'],
            prompts
        )
        
        # 为每个视图生成GradCAM
        for view_idx, (view_image, view_name) in enumerate([(frontal_image, 'frontal'), (lateral_image, 'lateral')]):
            # 生成CAM
            cam = grad_cam.generate_cam(view_image.unsqueeze(0))
            
            # 可视化并保存结果
            save_path = os.path.join('logs/gradcam', f'gradcam_{disease.replace(" ", "_")}_{view_name}.png')
            show_cam_on_image(
                view_image.permute(1, 2, 0).numpy(),
                cam,
                f"Disease: {disease}\nView: {view_name}",
                save_path
            )
            print(f"GradCAM result saved to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_index', type=int, default=0,
                      help='Index of the sample to visualize')
    args = parser.parse_args()
    main(args) 
