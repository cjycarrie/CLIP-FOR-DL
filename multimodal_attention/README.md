Updates on Multimodal Attention & Zero-Shot Prediction

1. Multimodal Attention Integration

A multimodal attention mechanism is introduced into the model pipeline. 
The multimodal attention module is now invoked in both the training and inference stages. The model checks for the presence of a `multimodal_attention` module in the `models` dictionary. 
In `train.py`, after extracting image and text features, the features are fused using the `multimodal_attention` module before loss computation.
In `disease_analysis.py`, the `predict_zero_shot` function applies the multimodal attention module (if present) to enhance image features before making predictions.

2. Enhanced Zero-Shot Prediction for Multi-Label Tasks

The zero-shot prediction pipeline has been significantly improved to better support multi-label disease classification:

- Dynamic Thresholding:
  - The `predict_zero_shot` function now supports per-disease dynamic thresholds, allowing for more flexible and accurate multi-label predictions.
  - Thresholds can be determined based on validation set statistics, improving F1 scores for rare and common diseases alike.

- Multi-View Fusion: 
  - For datasets with multiple image views (e.g., frontal and lateral X-rays), predictions from each view are merged using a weighted strategy, giving higher importance to the frontal view.

- Temperature Scaling:
  - Similarity scores between image and text features are now temperature-scaled for better probability calibration.

3. Code Changes Overview

- `train.py`: 
  - Integrated multimodal attention in the training loop.
  - Loss and prediction calculations now use enhanced features.

- `disease_analysis.py`: 
  - `predict_zero_shot` now supports multimodal attention, dynamic thresholds, and multi-view fusion.
  - Improved text feature extraction and prompt handling.

- `zero_shot_predict.py`:
  - Implements dynamic threshold determination and multi-view prediction merging for robust evaluation.

- `prepare_data.py`:
  - Added helper functions for dynamic thresholding and co-occurrence-based prediction adjustment.
