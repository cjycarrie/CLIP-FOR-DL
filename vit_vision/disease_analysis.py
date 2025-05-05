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

    stats = {}
    
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
            if idx == 0:  
                stats[disease]['first_position_count'] += 1
    
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    
    stats_df['percentage_as_first'] = (stats_df['first_position_count'] / stats_df['total_count']) * 100
    
    stats_df['frequency'] = stats_df['total_count']
    
    return stats_df

def create_rich_prompts(disease_stats):

    prompts = {}
    for disease, stats in disease_stats.iterrows():
        templates = []
        
        templates.extend([
            f"This chest X-ray shows {disease}.",
            f"The radiological findings indicate {disease}.",
            f"The image reveals characteristics of {disease}.",
            f"Diagnostic features of {disease} are present.",
            f"The X-ray demonstrates {disease}."
        ])
        
        if stats['frequency'] > 10: 
            templates.extend([
                f"This is a typical case of {disease}.",
                f"Clear radiological signs of {disease} are visible.",
                f"The X-ray shows classic features of {disease}."
            ])
        elif stats['frequency'] > 5: 
            templates.extend([
                f"This X-ray exhibits features consistent with {disease}.",
                f"Radiological patterns suggest {disease}."
            ])
        else:  
            templates.extend([
                f"This X-ray shows possible signs of {disease}.",
                f"Some features in this X-ray may indicate {disease}."
            ])
        
        if stats['percentage_as_first'] > 80: 
            templates.extend([
                f"The primary finding in this chest X-ray is {disease}.",
                f"This X-ray primarily shows {disease}."
            ])
        elif stats['percentage_as_first'] > 50: 
            templates.extend([
                f"One of the main findings in this X-ray is {disease}.",
                f"This X-ray shows significant evidence of {disease}."
            ])
        else:  
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

    inputs = tokenizer(
        findings,
        return_tensors='pt',
        padding='max_length',
        max_length=MODEL_CONFIG['max_text_length'],
        truncation=True
    ).to(device)
    
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

    prompts = []
    for disease in diseases:
        if disease == "Normal":
            prompt = "This is a normal chest X-ray without any significant findings."
        else:
            prompt = f"This chest X-ray shows {disease}."
        prompts.append(prompt)
    
    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding='max_length',
        max_length=MODEL_CONFIG['max_text_length'],
        truncation=True
    ).to(device)
    
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

    disease_prompts = prompts.get(disease, [f"This is a chest X-ray showing {disease}."])
    
    encoded = tokenizer(
        disease_prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=MODEL_CONFIG['max_text_length']
    ).to(device)

    with torch.no_grad():
        outputs = text_model(**encoded)
        text_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_proj = text_projector(text_emb)
        text_proj = F.normalize(text_proj, dim=-1)

    return text_proj.mean(dim=0, keepdim=True)

def predict_multilabel(image_features, text_features, threshold=0.5):

    similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']

    probabilities = torch.sigmoid(similarities)

    predictions = (probabilities > threshold).float()
    return predictions

def get_disease_cooccurrence(df):

    all_diseases = set()
    for problems in df['Problems'].dropna():
        diseases = [d.strip() for d in problems.split(';')]
        all_diseases.update(diseases)
    

    cooccurrence = pd.DataFrame(0, 
                              index=list(all_diseases),
                              columns=list(all_diseases))
    

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

    for name, model in models.items():
        if hasattr(model, 'eval'):
            model.eval()
    
    with torch.no_grad():

        is_batch = isinstance(images, torch.Tensor) and images.dim() == 4
        if not is_batch:
            if isinstance(images, list):
                images = images[0]
            images = images.unsqueeze(0).to(DEVICE)
        elif not images.is_cuda and DEVICE == 'cuda':
            images = images.to(DEVICE)
        

        image_embeddings = models['resnet'](images)
        image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)
        image_features = models['image_projector'](image_embeddings)
        image_features = F.normalize(image_features, dim=-1)
        

        text_features = get_prediction_text_features(
            disease_list,
            models['tokenizer'],
            models['text_model'],
            models['text_projector']
        )
        
 
        similarities = (image_features @ text_features.T) / MODEL_CONFIG['temperature']
        probabilities = F.softmax(similarities, dim=-1)

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

    if isinstance(true_labels, pd.DataFrame):

        y_true = np.array(true_labels['labels'].tolist())
    else:

        y_true = np.array(true_labels)
    

    y_pred = np.zeros((len(predictions), len(disease_list)))
    for i, preds in enumerate(predictions):
        for pred in preds:
            if pred in disease_list:
                y_pred[i, disease_list.index(pred)] = 1
    

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    

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

    prompts = {}
    

    for _, row in df.iterrows():
        if pd.isna(row['Problems']) or pd.isna(row['Findings']):
            continue
            
        diseases = [d.strip() for d in row['Problems'].split(';')]
        findings = row['Findings'].strip()
        

        for disease in diseases:
            if disease not in prompts:
                prompts[disease] = []
            

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

    all_features = []
    
    for disease in diseases:

        disease_prompts = prompts.get(disease, [f"This is a chest X-ray showing {disease}."])
        

        inputs = tokenizer(
            disease_prompts,
            return_tensors='pt',
            padding='max_length',
            max_length=MODEL_CONFIG['max_text_length'],
            truncation=True
        ).to(device)
        

        with torch.no_grad():
            outputs = text_model(**inputs)
            text_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            text_features = text_projector(text_embeddings)
            text_features = F.normalize(text_features, dim=-1)

            avg_feature = text_features.mean(dim=0, keepdim=True)
            all_features.append(avg_feature)
    
    return torch.cat(all_features, dim=0) 
