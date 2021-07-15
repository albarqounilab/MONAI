"""
Computes Area Under the Precision-Recall Curve (AUPRC). Referring to:
`sklearn.metrics.precision_recall_curve, average_precision_score
The input `y_pred` and `y` can be a list of `channel-first` Tensor or a `batch-first` Tensor.
"""
from sklearn.metrics import precision_recall_curve, average_precision_score


def compute_prc(predictions, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels.astype(int), predictions)
    auprc = average_precision_score(labels.astype(int), predictions)
    return auprc, precisions, recalls, thresholds
