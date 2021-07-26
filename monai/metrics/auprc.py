"""
Computes Area Under the Precision-Recall Curve (AUPRC). Referring to:
`sklearn.metrics.precision_recall_curve, average_precision_score
The input `y_pred` and `y` can be a list of `channel-first` Tensor or a `batch-first` Tensor.
"""
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
from .metric import IterationMetric


class PRCMetric(IterationMetric):
    """
    Computes Area Under the Precision-Recall Curve (AUPRC). Referring to:
    `sklearn.metrics.precision_recall_curve, average_precision_score
    The input `y_pred` and `y` can be a list of `channel-first` Tensor or a `batch-first` Tensor.
    """

    def __init__(self) -> None:
        super().__init__()

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor = None):  # type: ignore
        return compute_prc(y_pred, y)


def compute_prc(y_pred, y):
    y_pred = y_pred.cpu().detach().numpy().flatten()
    y = y.cpu().detach().numpy().flatten()
    print(type(y_pred), type(y), y_pred.shape, y.shape)
    precisions, recalls, thresholds = precision_recall_curve(y.astype(int), y_pred)
    auprc = average_precision_score(y.astype(int), y_pred)
    return auprc, precisions, recalls, thresholds






