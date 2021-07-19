from torch.nn.modules.loss import _Loss
import torch


class CosineEmbeddingLoss(_Loss):
    """
    Wrapper for pytorch CosineEmbeddingLoss
    see torch.nn.CosineEmbeddingLoss(margin=0.0, reduce=None, reduction='mean')

    """

    def __init__(
        self,
        margin: float = 0.0,
        reduction: str = 'mean') -> None:
        """
        Args
            margin: float, default: 0.0
                Should be a number from -1 to 1; 0 to 0.5 is suggested.
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.cs = torch.nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)

    def forward(self, input1: torch.Tensor, input2:  torch.Tensor, target: torch.Tensor):
        """
        Args:
            input1: (N,D),
                where N is the batch size and D is the embedding dimension.
            input2 (N,D),
                same shape as input1.
            target: (N).
                with values 1 or -1. This is used for measuring whether two inputs are similar or dissimilar
        """
        return self.cs(input1, input2, target)
