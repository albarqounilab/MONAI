import torch

from torch.nn import Module
from torch.distributions import Independent, Normal, kl_divergence


class StatisticsAlignmentLoss(Module):
    """
    Statistics alignment loss function for FedBias.
    """
    def __init__(self, device: torch.device, kl_direction: str = 'forward') -> None:
        """
        Initialization of StatisticsAlignmentLoss.

        :param device: torch.device
            Device on which to perform computations: torch.device('cuda:0') or torch.device('cpu').
        :param kl_direction: str
            Direction of the KL divergence: 'forward' or 'reverse'.
            See https://agustinus.kristia.de/techblog/2016/12/21/forward-reverse-kl/ for further information.
        """
        super().__init__()
        self.device = device
        self.kl_direction = kl_direction

    def forward(self, client_name: str, current_mean: torch.FloatTensor, current_std: torch.FloatTensor,
                client_stats: dict) -> torch.FloatTensor:
        """
        Computes the alignment loss.

        :param client_name: str
            Name of the current client.
        :param current_mean: torch.FloatTensor
            Mean vector of shape (D,) of features from model of client_name.
        :param current_std: torch.FloatTensor
            Standard deviation vector of shape (D,) of features from model of client_name.
        :param client_stats: dict
            Dict containing statistics (mean vector and standard deviation vector) for all clients with entries:
            client_name: (mean, std)
        :return:
            loss: torch.FloatTensor
                Loss value.
        """
        alignment_loss = torch.tensor(0., device=self.device)

        # Create multivariate normal distribution of features from client_name
        current_dist = Independent(Normal(current_mean, current_std), 1)
        for name in client_stats:
            # Only align to feature statistics of other clients
            if name != client_name:
                mean, std = client_stats[name]
                mean, std = mean.to(self.device), std.to(self.device)

                # Create multivariate normal distribution of features from other client
                dist = Independent(Normal(mean, std), 1)
                if self.kl_direction == 'forward':
                    alignment_loss = alignment_loss + kl_divergence(dist, current_dist)
                elif self.kl_direction == 'reverse':
                    alignment_loss = alignment_loss + kl_divergence(current_dist, dist)

        alignment_loss = alignment_loss / (len(client_stats) - 1)
        return alignment_loss
