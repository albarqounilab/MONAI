import numpy as np
"""
Computes DICE score. Referring to:
The input `y_pred` and `y` can be a list of `channel-first` Tensor or a `batch-first` Tensor.
"""
import torch
from .metric import IterationMetric


class DiceScore(IterationMetric):
    """
    Computes Area Under the Precision-Recall Curve (AUPRC). Referring to:
    `sklearn.metrics.precision_recall_curve, average_precision_score
    The input `y_pred` and `y` can be a list of `channel-first` Tensor or a `batch-first` Tensor.
    """

    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor = None):  # type: ignore
        return compute_dice(y_pred, y)


def compute_dice(y_pred, y):
    y_pred = y_pred.flatten()
    y = y.flatten()
    psum = torch.sum(y_pred)
    gsum = torch.sum(y)
    if gsum <= 1:
        return np.nan
    pgsum = torch.sum(torch.multiply(y_pred, y))
    score = (2 * pgsum) / (psum + gsum)
    return score