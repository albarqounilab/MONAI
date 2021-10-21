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
    def forward(self, probs: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes the entropy loss.

        :param probs: torch.FloatTensor
            Tensor with class probabilities of shape (N, D).
        :return:
        """
        if torch.any(probs < 0) or torch.any(probs > 1):
            raise Exception('Entropy loss takes probabilities 0 <= probs <= 1')

        # For numerical stability while taking log
        probs = probs + 1e-16
        entropy = torch.mean(torch.sum(probs * torch.log(probs), dim=1))
        return entropy
