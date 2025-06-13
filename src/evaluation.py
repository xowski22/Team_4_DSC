import torch
import numpy as np
from typing import Union, List


def map_at_10(predictions: Union[List[List[int]], torch.Tensor], 
              ground_truth: Union[List[List[int]], torch.Tensor]) -> float:
    """Calculate MAP@10 for recommendation model outputs."""
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Handle single-item ground truth
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.reshape(-1, 1)
    
    ap_scores = []
    for pred, truth in zip(predictions, ground_truth):
        # Remove padding/zeros from truth
        truth = truth[truth > 0] if hasattr(truth, '__len__') else [truth]
        if len(truth) == 0:
            ap_scores.append(0.0)
            continue
            
        # Calculate AP@10
        relevant_mask = np.isin(pred[:10], truth)
        if not relevant_mask.any():
            ap_scores.append(0.0)
            continue
            
        precisions = np.cumsum(relevant_mask) / np.arange(1, len(relevant_mask) + 1)
        ap_scores.append(np.mean(precisions[relevant_mask]))
    
    return np.mean(ap_scores)


