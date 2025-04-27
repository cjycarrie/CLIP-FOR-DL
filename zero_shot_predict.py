import os
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from config import MODEL_CONFIG, TRAINING_CONFIG, LOG_CONFIG, DEVICE, DATA_PATH
from prepare_data import load_data
from disease_analysis import predict_zero_shot, evaluate_predictions, create_rich_prompts, analyze_disease_distribution
from visualization import visualize_predictions
from train import initialize_models

def main():
    # 设置日志
    os.makedirs(LOG_CONFIG['log_dir'], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_CONFIG['log_dir'], 'zero_shot.log')),
            logging.StreamHandler()
        ]
    )

    # 加载数据
    logging.info("Loading data...")
    train_loader, val_loader, disease_list, reports_df = load_data()

    # 初始化模型
    models = initialize_models(DEVICE)

    # 加载训练好的模型
    logging.info("Loading trained model...")
    checkpoint_path = os.path.join(LOG_CONFIG['checkpoint_dir'], 'model_best.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    # 加载模型参数
    for name, model in models.items():
        if name in checkpoint['models']:
            try:
                model.load_state_dict(checkpoint['models'][name])
                logging.info(f"Loaded model: {name}")
            except Exception as e:
                logging.warning(f"Failed to load model {name}: {str(e)}")

    # 分析疾病分布并创建丰富的提示
    disease_stats = analyze_disease_distribution(reports_df)
    prompts = create_rich_prompts(disease_stats)

    # 进行零样本预测
    logging.info("Performing zero-shot prediction...")
    all_predictions = []
    all_scores = []
    all_images = []
    all_true_labels = []
    
    # 设置模型为评估模式
    for model in models.values():
        if hasattr(model, 'eval'):
            model.eval()

    with torch.no_grad():
        for batch_idx, (images, labels, findings, view_types) in enumerate(val_loader):
            # 处理多视图输入 [batch_size, 2, 3, 224, 224] -> [batch_size * 2, 3, 224, 224]
            batch_size = images.size(0)
            images = images.view(-1, 3, 224, 224).to(DEVICE)
            
            # 获取每个视图的预测
            batch_predictions, batch_scores = predict_zero_shot(
                images, 
                models, 
                disease_list, 
                top_k=3,
                prompts=prompts,
                use_enhanced_prompts=True
            )
            
            # 重塑预测结果以合并多个视图的预测
            view_predictions = [batch_predictions[i:i+2] for i in range(0, len(batch_predictions), 2)]
            view_scores = [batch_scores[i:i+2] for i in range(0, len(batch_scores), 2)]
            
            # 合并多视图预测（取两个视图预测的并集）
            merged_predictions = []
            merged_scores = []
            for view_pred, view_score in zip(view_predictions, view_scores):
                # 合并两个视图的预测
                combined_pred = list(set(view_pred[0] + view_pred[1]))[:3]  # 取前3个预测
                # 对应的分数（取最高分）
                combined_scores = []
                for disease in combined_pred:
                    max_score = max(
                        view_score[0][view_pred[0].index(disease)] if disease in view_pred[0] else 0,
                        view_score[1][view_pred[1].index(disease)] if disease in view_pred[1] else 0
                    )
                    combined_scores.append(max_score)
                
                merged_predictions.append(combined_pred)
                merged_scores.append(combined_scores)
            
            # 将预测结果转换为多标签格式
            batch_pred_matrix = np.zeros((batch_size, len(disease_list)))
            for i, preds in enumerate(merged_predictions):
                for pred in preds:
                    if pred in disease_list:
                        batch_pred_matrix[i, disease_list.index(pred)] = 1
            
            all_predictions.append(batch_pred_matrix)
            all_scores.extend(merged_scores)
            all_true_labels.append(labels.cpu().numpy())
            
            # 保存一些图像用于可视化
            if batch_idx < 5:  # 只保存前5个批次的图像
                all_images.extend(images[:batch_size].cpu().numpy())  # 只保存第一个视图

    # 合并所有批次的预测和标签
    all_predictions = np.vstack(all_predictions)
    all_true_labels = np.vstack(all_true_labels)

    # 评估预测结果
    logging.info("Evaluating predictions...")
    metrics = evaluate_predictions(all_predictions, all_true_labels, disease_list)

    # 打印评估结果
    logging.info("\nZero-shot Prediction Results:")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logging.info(f"Micro F1: {metrics['micro_f1']:.4f}")
    logging.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    # 保存结果
    results_df = pd.DataFrame(metrics['classification_report']).transpose()
    results_df.to_csv(os.path.join(LOG_CONFIG['log_dir'], 'zero_shot_results.csv'))

    # 可视化一些预测结果
    if len(all_images) > 0:
        visualize_predictions(
            all_images[:5], 
            [[disease_list[j] for j in range(len(disease_list)) if all_predictions[i][j] > 0] for i in range(5)],
            all_scores[:5], 
            disease_list,
            save_dir=os.path.join(LOG_CONFIG['log_dir'], 'zero_shot_predictions')
        )

if __name__ == '__main__':
    main() 