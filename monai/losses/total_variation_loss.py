"""
Total Variation Loss
"""
import torch

from torch.nn.modules.loss import _Loss


class TotalVariationLoss(_Loss):
    """
    Entropy loss = negative entropy.
    """
    def forward(self, x):
        """
        Computes the entropy loss.

        :param x: torch.FloatTensor
            Input Image of shape (B, C, W, H)
        :return:
        """
        tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return tv_loss
