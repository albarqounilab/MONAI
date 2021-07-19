from torch.nn.modules.loss import _Loss
import torch


class L1Loss(_Loss):
    """
    Wrapper for pytorch L1Loss
    see torch.nn.L1Loss(reduction='mean')

    """

    def __init__(
        self,
        reduction: str = 'mean') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.reduction = reduction
        self.l1 = torch.nn.L1Loss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        return self.l1(input, target)
