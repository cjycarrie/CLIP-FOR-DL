## Project Overview

The primary objective of this project is to evaluate the performance of **Vision Transformers (ViT)** as a replacement for ResNet-based backbones in a CLIP-style image-text learning setup for multi-label classification in chest X-ray images.

## Notable Files

- **`chestxray_CLIP_vitvision.ipynb`**:  
  Based on 02_chestxray_CLIP.ipynb, in which the original ResNet image encoder was replaced with a Vision Transformer (ViT) as an exploratory attempt. However, the introduction of ViT did not lead to a significant improvement in accuracy, and thus this approach has currently been abandoned.

- **Other scripts and modules**:  
  These are modified from the original "end-to-end multi-label image-text learning" approach. They incorporate:
  - A ViT-based image encoder
  - An updated Grad-CAM visualization method compatible with ViT
  
  Despite the architectural changes, the overall performance **did not improve significantly**, and **further research and experimentation are required** to enhance results.

## Notes

- All experiments were conducted on the ChestXray dataset.
- Accuracy remains suboptimal with ViT-based models in the current implementation.
- Future directions may include:
  - Better ViT pretraining strategies
  - More effective multimodal fusion mechanisms
  - Advanced interpretability techniques beyond Grad-CAM

