"""
Code adapted from https://github.com/human-analysis/MaxEnt-ARL/blob/master/loss/entropy.py (Roy et al. [1]).

[1] Proteek Chandan Roy, Vishnu Naresh Boddeti. Mitigating Information Leakage in Image Representations:
    A Maximum Entropy Approach. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    2019.
"""
import torch

from torch.nn.modules.loss import _Loss


class EntropyLoss(_Loss):
    """
    Entropy loss = negative entropy.
    """
    def forward(self, prob: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes the entropy loss.

        :param prob: torch.FloatTensor
            Tensor with class probabilities of shape (N, D).
        :return:
        """
        if torch.any(prob < 0) or torch.any(prob > 1):
            raise Exception('Entropy Loss takes probabilities 0 <= prob <= 1')

        # For numerical stability while taking log
        prob = prob + 1e-16
        entropy = torch.mean(torch.sum(prob * torch.log(prob), dim=1))
        return entropy
