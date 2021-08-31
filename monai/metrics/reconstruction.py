import torch
from .metric import IterationMetric
from skimage.metrics import structural_similarity as ssim


class SSIMMetric(IterationMetric):
    """Compute Structural similarity index measure (SSIM)


    More info: https://en.wikipedia.org/wiki/Structural_similarity

    See:
    skimage.metrics.structural_similarity

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued.
    """

    def __init__(
        self,
        data_range,
        reduction
    ) -> None:
        self.data_range = data_range
        self.reduction = reduction
        super().__init__()

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor = None):
        y_pred = y_pred.float().numpy()
        y = y.float().numpy()
        ssim_val = ssim(y_pred, y, data_range=self.data_range, reduction=self.reduction)
        return ssim_val