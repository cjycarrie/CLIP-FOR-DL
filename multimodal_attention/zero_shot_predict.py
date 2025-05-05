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
from sklearn.metrics import f1_score

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

    # 动态阈值：根据验证集的预测分布来确定
    thresholds = {disease: 0.3 for disease in disease_list}  # 初始阈值
    
    # 第一遍：收集预测分数分布
    logging.info("Collecting prediction score distribution...")
    all_disease_scores = {disease: [] for disease in disease_list}
    all_disease_labels = {disease: [] for disease in disease_list}
    
    with torch.no_grad():
        for batch_idx, (images, labels, findings, view_types) in enumerate(val_loader):
            if batch_idx >= len(val_loader) // 4:  # 只使用25%的验证集来确定阈值
                break
                
            batch_size = images.size(0)
            images = images.view(-1, 3, 224, 224).to(DEVICE)
            
            # 获取每个视图的预测
            batch_predictions, batch_scores = predict_zero_shot(
                images, 
                models, 
                disease_list,
                threshold=0.0,  # 不使用阈值筛选
                prompts=prompts,
                use_enhanced_prompts=True
            )
            
            # 重塑预测结果以合并多个视图的预测
            view_scores = [batch_scores[i:i+2] for i in range(0, len(batch_scores), 2)]
            
            # 收集每个疾病的预测分数和真实标签
            for i in range(batch_size):
                true_labels = labels[i].cpu().numpy()
                # 合并两个视图的分数，取最大值
                scores_view1 = view_scores[i][0]
                scores_view2 = view_scores[i][1]
                max_scores = []
                for j in range(len(scores_view1)):
                    max_score = max(scores_view1[j], scores_view2[j])
                    max_scores.append(max_score)
                
                # 保存每个疾病的分数和标签
                for disease_idx, disease in enumerate(disease_list):
                    all_disease_scores[disease].append(max_scores[disease_idx])
                    all_disease_labels[disease].append(true_labels[disease_idx])
    
    # 根据分数分布和真实标签确定每个疾病的最优阈值
    logging.info("Determining optimal thresholds...")
    for disease in disease_list:
        scores = np.array(all_disease_scores[disease])
        labels = np.array(all_disease_labels[disease])
        if len(scores) > 0:
            # 计算正样本和负样本的分数分布
            pos_scores = scores[labels == 1]
            neg_scores = scores[labels == 0]
            
            # 如果没有正样本，使用较高的阈值
            if len(pos_scores) == 0:
                thresholds[disease] = 0.8
                logging.info(f"{disease}: No positive samples, using default threshold = 0.8")
                continue
            
            # 如果没有负样本，使用较低的阈值
            if len(neg_scores) == 0:
                thresholds[disease] = 0.2
                logging.info(f"{disease}: No negative samples, using default threshold = 0.2")
                continue
            
            # 计算正负样本的统计信息
            pos_mean = np.mean(pos_scores)
            pos_std = np.std(pos_scores)
            neg_mean = np.mean(neg_scores)
            neg_std = np.std(neg_scores)
            
            # 尝试不同的阈值
            best_f1 = 0
            best_threshold = 0.5
            
            # 根据分布确定搜索范围
            min_threshold = max(0.1, neg_mean - neg_std)
            max_threshold = min(0.9, pos_mean + pos_std)
            
            for threshold in np.linspace(min_threshold, max_threshold, 20):
                predictions = (scores >= threshold).astype(int)
                f1 = f1_score(labels, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            thresholds[disease] = best_threshold
            logging.info(f"{disease}:")
            logging.info(f"  Positive samples: mean = {pos_mean:.3f}, std = {pos_std:.3f}")
            logging.info(f"  Negative samples: mean = {neg_mean:.3f}, std = {neg_std:.3f}")
            logging.info(f"  Best threshold = {best_threshold:.3f}, F1 = {best_f1:.3f}")
    
    # 第二遍：使用确定的阈值进行正式预测
    with torch.no_grad():
        for batch_idx, (images, labels, findings, view_types) in enumerate(val_loader):
            # 处理多视图输入
            batch_size = images.size(0)
            images = images.view(-1, 3, 224, 224).to(DEVICE)
            
            # 获取每个视图的预测
            batch_predictions, batch_scores = predict_zero_shot(
                images, 
                models, 
                disease_list,
                threshold=thresholds,  # 使用每个疾病的特定阈值
                prompts=prompts,
                use_enhanced_prompts=True
            )
            
            # 重塑预测结果以合并多个视图的预测
            view_predictions = [batch_predictions[i:i+2] for i in range(0, len(batch_predictions), 2)]
            view_scores = [batch_scores[i:i+2] for i in range(0, len(batch_scores), 2)]
            
            # 使用加权平均合并多视图预测
            merged_predictions = []
            merged_scores = []
            for view_pred, view_score in zip(view_predictions, view_scores):
                # 计算每个疾病的加权分数
                disease_scores = {}
                for view_idx, (preds, scores) in enumerate(zip(view_pred, view_score)):
                    weight = 1.0 if view_idx == 0 else 0.8  # 给予正面视图更高的权重
                    for pred, score in zip(preds, scores):
                        if pred not in disease_scores:
                            disease_scores[pred] = 0
                        disease_scores[pred] = max(disease_scores[pred], score * weight)
                
                # 根据动态阈值筛选预测
                filtered_predictions = []
                filtered_scores = []
                for disease, score in disease_scores.items():
                    if score >= thresholds[disease]:
                        filtered_predictions.append(disease)
                        filtered_scores.append(score)
                
                # 如果没有预测超过阈值，选择最高的分数
                if not filtered_predictions:
                    max_disease = max(disease_scores.items(), key=lambda x: x[1])
                    filtered_predictions = [max_disease[0]]
                    filtered_scores = [max_disease[1]]
                
                merged_predictions.append(filtered_predictions)
                merged_scores.append(filtered_scores)
            
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
    logging.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logging.info(f"Micro F1: {metrics['micro_f1']:.4f}")
    logging.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # 打印每个疾病的详细指标
    logging.info("\nPer-class Performance:")
    for disease in disease_list:
        metrics_dict = metrics['per_class'][disease]
        logging.info(f"\n{disease}:")
        logging.info(f"  Precision: {metrics_dict['precision']:.4f}")
        logging.info(f"  Recall: {metrics_dict['recall']:.4f}")
        logging.info(f"  F1-score: {metrics_dict['f1']:.4f}")

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
