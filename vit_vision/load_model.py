import torch
from config import MODEL_CONFIG, DEVICE
from train import initialize_models
import torch.nn.functional as F

def get_text_features(disease_list, tokenizer, text_model, text_projector):

    prompts = [f"a chest x-ray of {disease.lower()}" for disease in disease_list]
    
    text_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    text_tokens = {k: v.to(DEVICE) for k, v in text_tokens.items()}
    
    text_embeddings = text_model(**text_tokens)
    text_embeddings = text_embeddings.last_hidden_state[:, 0, :]
    text_features = text_projector(text_embeddings)
    
    text_features = F.normalize(text_features, dim=-1)
    
    return text_features

def load_trained_model(checkpoint_path='checkpoints/model_best.pth'):

    models = initialize_models(DEVICE)
    

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
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
    
    for name, model in models.items():
        if name != 'tokenizer' and hasattr(model, 'eval'):
            model.eval()
    
    return models

def get_model_predictions(models, images, disease_list):

    with torch.no_grad():
        try:
            device = next(models['resnet'].parameters()).device
            
            images = images.to(device)
            
            views = images.size(1)
            all_view_features = []
            
            for v in range(views):
                view_images = images[:, v]
                view_embeddings = models['resnet'](view_images)
                view_embeddings = view_embeddings.view(view_embeddings.size(0), -1)
                view_features = models['image_projector'](view_embeddings)
                all_view_features.append(view_features)
            
            image_features = models['view_fusion'](all_view_features[0], all_view_features[1])
            
            text_features = get_text_features(
                disease_list,
                models['tokenizer'],
                models['text_model'],
                models['text_projector']
            )
            
            if text_features.device != device:
                text_features = text_features.to(device)
            
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            similarities = image_features @ text_features.T
            probabilities = torch.sigmoid(similarities * 4.0)  
            predictions = (probabilities > 0.5).float()
            
            return predictions.cpu(), probabilities.cpu()
            
        except Exception as e:
            print(f"Error in get_model_predictions: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

if __name__ == "__main__":
    models = load_trained_model()
    if models is not None:
        print("\nModel loaded successfully and ready for inference!")
        print("\nExample usage:")
        print("1. Load your image data")
        print("2. Call get_model_predictions(models, images, disease_list)")
        print("3. Process the predictions as needed") 
