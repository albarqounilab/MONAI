from torch.nn.modules.loss import _Loss
import torch
import numpy as np
from monai.utils.misc import generate_tensor
from monai.transforms import Compose, LoadImage, AsChannelFirst, NormalizeIntensity, ScaleIntensity, SpatialPad, Resize, \
    ToTensor

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


class AtlasPriorLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.

    """

    def __init__(self, atlas, dim_c: int = 1) -> None:
        """
        Args
            dim_c: int
                the number of clusters
        """
        super().__init__()
        self.atlas = atlas
        # ld = LoadImage(image_only=True)
        # atlas_ = ld(seg_path)
        # mask_ = ld(mask_path)
        # atlas_np = np.concatenate([atlas_, 1-mask_[:, :, :, np.newaxis]], axis=-1)
        # self.atlas = torch.from_numpy(np.transpose(atlas_np, (0, 3, 1, 2)))
        self.loss = torch.nn.CrossEntropyLoss()
        # self.loss = torch.nn.L1Loss()
        # self.loss = torch.nn.MSELoss()
        self.dim_c = dim_c

    def forward(self, pc: torch.Tensor, atlas=None):
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
        if atlas is not None:
            self.atlas = atlas
        mean_con_loss = self.loss(pc, self.atlas)
        return mean_con_loss


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
        con_prior_loss = torch.sum(torch.squeeze(torch.matmul(KL, torch.unsqueeze(pc, -1)), -1), [1, 2, 3])
        mean_con_loss = torch.maximum(torch.cuda.FloatTensor([[100]]), torch.mean(con_prior_loss))

        return mean_con_loss


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
        mean_w_loss = torch.maximum(torch.cuda.FloatTensor([[2]]), torch.mean(w_loss))

        return mean_w_loss


class CPriorLoss(_Loss):
    """
    Wrapper for pytorch GMVAE Loss
    see You, Suhang, et al. "Unsupervised lesion detection via image restoration with a normative prior." International Conference on Medical Imaging with Deep Learning. PMLR, 2019.

    """

    def __init__(self, dim_c: int = 6, c_lambda: int = 0.5) -> None:
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
        mean_c_loss = torch.mean(c_loss)#torch.maximum(torch.cuda.FloatTensor([[700]]), torch.mean(c_loss)) # cutoff value

        return mean_c_loss