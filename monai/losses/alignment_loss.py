import torch

from torch.nn import Module
from torch.distributions import MultivariateNormal, kl_divergence


class StatisticsAlignmentLoss(Module):
    """
    Statistics alignment loss function for FedBias.
    """
    def __init__(self, device: torch.device) -> None:
        """
        Initialization of StatisticsAlignmentLoss.

        :param device: torch.device
            Device on which to perform computations: torch.device('cuda:0') or torch.device('cpu').
        """
        super().__init__()
        self.device = device

    def forward(self, client_name: str, current_mean: torch.FloatTensor, current_cov: torch.FloatTensor,
                client_stats: dict) -> torch.FloatTensor:
        """
        Computes the alignment loss.

        :param client_name: str
            Name of the current client.
        :param current_mean: torch.FloatTensor
            Mean vector of shape (D,) of features from model of client_name.
        :param current_cov: torch.FloatTensor
            Covariance matrix of shape (D, D) of features from model of client_name.
        :param client_stats: dict
            Dict containing statistics (mean vector and covariance matrix) for all clients with entries:
            client_name: (mean, cov)
        :return:
            loss: torch.FloatTensor
                Loss value.
        """
        # Create multivariate normal distribution of features from client_name
        current_dist = MultivariateNormal(current_mean, current_cov)
        alignment_loss = 0
        for name in client_stats:
            # Only align to feature statistics of other clients
            if name != client_name:
                mean, cov = client_stats[name]
                mean, cov = mean.to(self.device), cov.to(self.device)
                # Create multivariate normal distribution of features from other client
                dist = MultivariateNormal(mean, cov)
                # NOTE: Use forward KL divergence (https://agustinus.kristia.de/techblog/2016/12/21/forward-reverse-kl/)
                alignment_loss = alignment_loss + kl_divergence(dist, current_dist)
        alignment_loss = alignment_loss / (len(client_stats) - 1)
        return alignment_loss
