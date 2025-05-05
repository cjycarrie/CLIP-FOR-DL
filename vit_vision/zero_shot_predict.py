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
    os.makedirs(LOG_CONFIG['log_dir'], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_CONFIG['log_dir'], 'zero_shot.log')),
            logging.StreamHandler()
        ]
    )

    logging.info("Loading data...")
    train_loader, val_loader, disease_list, reports_df = load_data()

    models = initialize_models(DEVICE)

    logging.info("Loading trained model...")
    checkpoint_path = os.path.join(LOG_CONFIG['checkpoint_dir'], 'model_best.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    for name, model in models.items():
        if name in checkpoint['models']:
            try:
                model.load_state_dict(checkpoint['models'][name])
                logging.info(f"Loaded model: {name}")
            except Exception as e:
                logging.warning(f"Failed to load model {name}: {str(e)}")

    disease_stats = analyze_disease_distribution(reports_df)
    prompts = create_rich_prompts(disease_stats)

    logging.info("Performing zero-shot prediction...")
    all_predictions = []
    all_scores = []
    all_images = []
    all_true_labels = []
    
    for model in models.values():
        if hasattr(model, 'eval'):
            model.eval()

    with torch.no_grad():
        for batch_idx, (images, labels, findings, view_types) in enumerate(val_loader):
            batch_size = images.size(0)
            images = images.view(-1, 3, 224, 224).to(DEVICE)
            
            batch_predictions, batch_scores = predict_zero_shot(
                images, 
                models, 
                disease_list, 
                top_k=3,
                prompts=prompts,
                use_enhanced_prompts=True
            )
            
            view_predictions = [batch_predictions[i:i+2] for i in range(0, len(batch_predictions), 2)]
            view_scores = [batch_scores[i:i+2] for i in range(0, len(batch_scores), 2)]
            
            merged_predictions = []
            merged_scores = []
            for view_pred, view_score in zip(view_predictions, view_scores):
                combined_pred = list(set(view_pred[0] + view_pred[1]))[:3]  
                combined_scores = []
                for disease in combined_pred:
                    max_score = max(
                        view_score[0][view_pred[0].index(disease)] if disease in view_pred[0] else 0,
                        view_score[1][view_pred[1].index(disease)] if disease in view_pred[1] else 0
                    )
                    combined_scores.append(max_score)
                
                merged_predictions.append(combined_pred)
                merged_scores.append(combined_scores)
            
            batch_pred_matrix = np.zeros((batch_size, len(disease_list)))
            for i, preds in enumerate(merged_predictions):
                for pred in preds:
                    if pred in disease_list:
                        batch_pred_matrix[i, disease_list.index(pred)] = 1
            
            all_predictions.append(batch_pred_matrix)
            all_scores.extend(merged_scores)
            all_true_labels.append(labels.cpu().numpy())
            
            if batch_idx < 5:  
                all_images.extend(images[:batch_size].cpu().numpy()) 

    all_predictions = np.vstack(all_predictions)
    all_true_labels = np.vstack(all_true_labels)

    logging.info("Evaluating predictions...")
    metrics = evaluate_predictions(all_predictions, all_true_labels, disease_list)

    logging.info("\nZero-shot Prediction Results:")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logging.info(f"Micro F1: {metrics['micro_f1']:.4f}")
    logging.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    results_df = pd.DataFrame(metrics['classification_report']).transpose()
    results_df.to_csv(os.path.join(LOG_CONFIG['log_dir'], 'zero_shot_results.csv'))

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
