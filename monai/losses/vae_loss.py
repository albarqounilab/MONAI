from torch.nn.modules.loss import _Loss
import torch
import numpy as np
from monai.utils.misc import generate_tensor


class ReconLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.

    """

    def __init__(self) -> None:
        """
        Args
        """
        super().__init__()

    @staticmethod
    def log_likelihood_inv(x, mu, logvarinv, axis=-1):
        return -0.5 * torch.sum(torch.log(torch.tensor(2 * np.pi)) - logvarinv, axis), -0.5 * torch.sum(
            torch.pow(torch.subtract(x, mu), 2) * torch.exp(logvarinv), axis)

    def forward(self, input_tensor: torch.Tensor, recon_mean: torch.Tensor, recon_log_sigma: torch.Tensor):
        """
        Args:
            w_mean: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
            w_log_sigma: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
        """
        p_loss_part1, p_loss_part2 = self.log_likelihood_inv(input_tensor, recon_mean, recon_log_sigma, axis=[1, 2, 3])
        p_loss_part1 = - p_loss_part1
        p_loss_part2 = - p_loss_part2
        p_loss = p_loss_part1 + p_loss_part2
        mean_p_loss = torch.mean(p_loss)
        return mean_p_loss


class BrainPriorLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.

    """

    def __init__(self, atlas_mean, atlas_var) -> None:
        """
        Args
            dim_c: int
                the number of clusters
        """
        super().__init__()
        self.atlas_mean = atlas_mean
        self.atlas_var = atlas_var
        self.b, self.c, self.h, self.w = self.atlas_mean.shape
        self.mu_atlas = self.atlas_mean.view(self.b, self.c, -1)
        self.logsigma_atlas = self.atlas_var.view(self.b, self.c, -1)
        self.invsigma_atlas = 1 / torch.exp(self.logsigma_atlas)

        self.constant = torch.squeeze(torch.sum(self.logsigma_atlas, dim=2, keepdim=True)) - self.h * self.w

    def forward(self, pred_mean: torch.Tensor, pred_var: torch.Tensor):
        """
        Args:
            pred_mean: tensor of size (N, C, H, W),
                with N = batch, C = dim_z/channels, H/W = height/weight of latent space
            pred_var: tensor of size (N, H, W, C),
                with N = batch, C = dim_z/channels, H/W = height/weight of latent space
        """

        mu_pred, logsigma_pred = pred_mean.view(self.b, self.c, -1), pred_var.view(self.b, self.c, -1)

        log_det_pred = torch.squeeze(torch.sum(logsigma_pred, dim=2, keepdim=True))
        trace = torch.squeeze(torch.sum(self.invsigma_atlas * torch.exp(logsigma_pred), dim=2, keepdim=True))
        third_term = torch.matmul(torch.transpose(torch.unsqueeze(self.mu_atlas - mu_pred, -1), 2, 3),
                     torch.unsqueeze(self.invsigma_atlas * (self.mu_atlas - mu_pred), -1))

        kld = torch.mean(0.5 * (self.constant - log_det_pred + trace + third_term))

        return torch.mean(kld)


class ConditionalPriorLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.

    """

    def __init__(self, dim_c: int = 4) -> None:
        """
        Args
            dim_c: int
                the number of clusters
        """
        super().__init__()
        self.dim_c = dim_c

    def forward(self, z_mean: torch.Tensor, z_log_sigma: torch.Tensor, z_wc_mean: torch.Tensor,
                z_wc_log_sigma: torch.Tensor, pc: torch.Tensor):
        """
        Args:
            z_mean: tensor of size (N, H, W, C),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
            z_log_sigma: tensor of size (N, H, W, C),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
            z_wc_mean: tensor of size (N, H, W, C),
                with N = nr slices, C = dim_z*dim_c, H/W = height/weight of latent space
            z_wc_log_sigma: tensor of size (N, H, W, C),
                with N = nr slices, C = dim_z*dim_c, H/W = height/weight of latent space
            pc: tensor of size (N, H, W, C),
                with N = nr slices, C = dim_z*dim_c, H/W = height/weight of latent space
        """
        z_mean = torch.tile(torch.unsqueeze(z_mean, -1), [1, 1, 1, 1, self.dim_c])
        z_log_sigma = torch.tile(torch.unsqueeze(z_log_sigma, -1), [1, 1, 1, 1, self.dim_c])
        d_mu_2 = torch.pow(torch.subtract(z_mean, z_wc_mean), 2)
        d_var = (torch.exp(z_log_sigma) + d_mu_2) * (torch.exp(z_wc_log_sigma) + 1e-6)
        d_logvar = -1 * (z_wc_log_sigma + z_log_sigma)
        KL = (d_var + d_logvar - 1) * 0.5
        mean_con_loss = torch.mean(torch.sum(torch.squeeze(torch.matmul(KL, torch.unsqueeze(pc, -1)), -1), [1, 2, 3]))
        # mean_con_loss = torch.maximum(torch.cuda.FloatTensor([[100]]), torch.mean(con_prior_loss)) # cutoff

        return mean_con_loss


class NormalPriorLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.
    """

    def __init__(self) -> None:
        """
        Args
        """
        super().__init__()

    def forward(self, z_mean: torch.Tensor, z_log_sigma: torch.Tensor):
        """
        Args:
            z_mean: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
            z_log_sigma: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
        """
        merge_dims = [i for i in range(len(z_mean.shape))]
        merge_dims = merge_dims[1:]
        z_loss = 0.5 * torch.sum(torch.pow(z_mean, 2) + torch.exp(z_log_sigma) - z_log_sigma - 1, merge_dims)
        mean_z_loss = torch.mean(z_loss)

        # mu = z_mean
        # log_var = z_log_sigma
        # mean_z_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        return mean_z_loss


class CPriorLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.

    """

    def __init__(self, dim_c: int = 4, c_lambda: int = 0.5) -> None:
        """
        Args
        """
        super().__init__()
        self.dim_c = dim_c
        self.c_lambda = c_lambda

    def forward(self, pc: torch.Tensor):
        """
        Args:
            pc: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z*dim_c, H/W = height/weight of latent space
        """
        closs1 = torch.sum(torch.multiply(pc, torch.log(pc * self.dim_c + 1e-8)), [3])
        c_lambda = generate_tensor(closs1, self.c_lambda)
        c_loss = torch.maximum(closs1, c_lambda)
        c_loss = torch.sum(c_loss, [1, 2])
        mean_c_loss = torch.mean(c_loss)
        #torch.maximum(torch.cuda.FloatTensor([[700]]), torch.mean(c_loss)) # cutoff value

        return mean_c_loss