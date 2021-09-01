from torch.nn.modules.loss import _Loss
import torch


class ConditionalPriorLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.

    """

    def __init__(self, dim_c: int = 1) -> None:
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
            z_mean: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
            z_log_sigma: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
            z_wc_mean: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z*dim_c, H/W = height/weight of latent space
            z_wc_log_sigma: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z*dim_c, H/W = height/weight of latent space
            pc: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z*dim_c, H/W = height/weight of latent space
        """
        # calculate KL for each cluster
        # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 ) here dim_c clusters, then we have batchsize * dim_z * dim_c
        # then [batchsize * dim_z* dim_c] * [batchsize * dim_c * 1]  = batchsize * dim_z * 1, squeeze it to batchsize * dim_z
        z_mean = torch.tile(torch.unsqueeze(z_mean, -1), [1, 1, 1, 1, self.dim_c])
        z_log_sigma = torch.tile(torch.unsqueeze(z_log_sigma, -1), [1, 1, 1, 1, self.dim_c])
        d_mu_2 = torch.pow(torch.subtract(z_mean, z_wc_mean), 2)
        d_var = (torch.exp(z_log_sigma) + d_mu_2) * (torch.exp(z_wc_log_sigma) + 1e-6)
        d_logvar = -1 * (z_wc_log_sigma + z_log_sigma)
        kl = (d_var + d_logvar - 1) * 0.5
        kl = kl.permute(0, 2, 3, 1, 4)
        con_prior_loss = torch.mean(torch.sum(torch.squeeze(torch.matmul(kl, torch.unsqueeze(pc, -1)), dim=4), [1, 2, 3]))
        return con_prior_loss


class WPriorLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.

    TODO: Replace 0 mean, 1 std gaussian with atlas tissue prior here
    """

    def __init__(self) -> None:
        """
        Args
        """
        super().__init__()

    def forward(self, w_mean: torch.Tensor, w_log_sigma: torch.Tensor):
        """
        Args:
            w_mean: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
            w_log_sigma: tensor of size (N, C, H, W),
                with N = nr slices, C = dim_z, H/W = height/weight of latent space
        """
        w_loss = 0.5 * torch.sum(torch.pow(w_mean, 2) + torch.exp(w_log_sigma) - w_log_sigma - 1, [1, 2, 3])
        mean_w_loss = torch.mean(torch.tensor(w_loss))
        return mean_w_loss


class CPriorLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.

    """

    def __init__(self, dim_c: int = 6, c_lambda: int = 1) -> None:
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
        closs1 = torch.sum(torch.multiply(pc, torch.log(torch.tensor(pc * self.dim_c + 1e-8))), [3])
        c_lambda = torch.cuda.FloatTensor(closs1.shape).fill_(self.c_lambda)
        c_loss = torch.maximum(closs1, c_lambda)
        c_loss = torch.mean(torch.sum(c_loss, [1, 2]))
        return c_loss