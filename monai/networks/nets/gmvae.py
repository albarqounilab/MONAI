import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
from monai.networks.blocks import Convolution
from monai.networks.layers.convutils import weights_init
from monai.utils.misc import generate_tensor
from monai.networks.nets.convolutional_autoencoders import *

__all__ = ["GaussianMixtureVariationalAutoEncoder", "GaussianMixtureVariationalAutoEncoderBig"]


class GaussianMixtureVariationalAutoEncoder(nn.Module):

    def __init__(self, dimensions: int, in_channels: int, dim_z: int, dim_w: int, dim_c: int, channels: Sequence[int],
                 out_ch: int, strides: Sequence[int], kernel_size=5, norm='batch', act='leakyrelu', bottleneck=False,
                 skip=False,norm_mean=0, norm_std=1):
        """
        Variational Auto-Encoder
        :param dimensions:
        :param in_channels:
        :param dim_z:
        :param dim_w:
        :param dim_c
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param bottleneck:
        :param skip:
        :param norm_mean:
        :param norm_std:
        """
        super(GaussianMixtureVariationalAutoEncoder, self).__init__()

        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.dim_c = dim_c

        self.encode = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                              kernel_size=kernel_size, norm=norm, act=act, name_prefix='ENC_')

        self.decode = Decoder(dimensions=dimensions, in_channels=in_channels, channels=channels, out_ch=out_ch,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final='sigmoid',
                              bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='DEC_')

        z_channels = channels[-1]
        # Z
        self.z_mu_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z, strides=1,
                                     kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='')
        self.z_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z,
                                            strides=1, kernel_size=1, conv_only=True, bias=True, norm='batch',
                                            act='relu', name='')
        self.z_log_sigma_conv.apply(weights_init)

        # W
        self.w_mu_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_w, strides=1,
                                     kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='')
        self.w_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_w,
                                            strides=1, kernel_size=1, conv_only=True, bias=True, norm='batch',
                                            act='relu', name='')
        self.w_log_sigma_conv.apply(weights_init)

        # Z_WC
        self.z_wc_conv = Convolution(dimensions=dimensions, in_channels=dim_w, out_channels=channels[0], strides=1,
                                     kernel_size=1, conv_only=False, bias=True, norm='batch', act='relu', name='')
        self.z_wc_mu_conv = Convolution(dimensions=dimensions, in_channels=channels[0], out_channels=dim_z * dim_w,
                                        strides=1, kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu')
        self.z_wc_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=channels[0],
                                               out_channels=dim_z * dim_w, strides=1, kernel_size=1, conv_only=True,
                                               bias=True, norm='batch', act='relu')
        self.z_wc_log_sigma_conv.apply(weights_init)

    def _latent_distribution(self, z_enc: torch):
        #  q(z|x)
        z_mean = self.z_mu_conv(z_enc)
        z_log_sigma = self.z_log_sigma_conv(z_enc)
        z = z_mean + torch.exp(0.5 * z_log_sigma) * generate_tensor(z_mean)

        # q(w|x)
        w_mean = self.w_mu_conv(z_enc)
        w_log_sigma = self.w_log_sigma_conv(z_enc)
        w = w_mean + torch.exp(0.5 * w_log_sigma) * generate_tensor(w_mean)

        # posterior p(z|w,c)
        w_enc = self.z_wc_conv(w)
        z_wc_mean = self.z_wc_mu_conv(w_enc)
        z_wc_log_sigma = self.z_wc_log_sigma_conv(w_enc)
        z_wc = z_wc_mean + torch.exp(z_wc_log_sigma) * generate_tensor(z_wc_mean)

        # prior p(c) network
        z_sample = torch.tile(z, [1, self.dim_c, 1, 1])
        pc_logit = -0.5 * torch.pow(torch.subtract(z_sample, z_wc_mean), 2) * torch.exp(z_wc_log_sigma) \
                   - z_wc_log_sigma + torch.log(torch.tensor(np.pi))
        pc = self.softMax(pc_logit)

        return {'z': z, 'w': w, 'z_wc': z_wc, 'pc': pc}

    def _skip_forward(self, x: torch):
        # encoding network
        down_samples = []
        z_enc = x
        for layer in self.encode:
            z_enc = layer(z_enc)
            if hasattr(layer, 'conv'):
                down_samples.append(z_enc)

        # latent
        return_dict = self._latent_distribution(z_enc)
        x_ = return_dict['z']

        # decoding network
        for i, layer in enumerate(self.decode):
            if hasattr(layer, 'conv'):
                if layer.conv.stride[0] == 2:
                    x_ = torch.cat((x_, down_samples[-i - 1]), 1)
            x_ = layer(x_)

        return x_, return_dict

    def forward(self,  x: torch):
        # encode
        z_enc = self.encode(x)

        #  latent
        return_dict = self._latent_distribution(z_enc)
        z = return_dict['z']

        # decode
        x_ = self.decode(z)
        return x_, return_dict


class GaussianMixtureVariationalAutoEncoderBig(nn.Module):

    def __init__(self, dimensions: int, in_channels: int, dim_z: int, dim_w: int, dim_c: int, channels: Sequence[int],
                 out_ch: int, strides: Sequence[int], kernel_size=5, norm='batch', act='leakyrelu', bottleneck=False,
                 skip=False, norm_mean=0, norm_std=1):
        """
        :param dimensions:
        :param in_channels:
        :param dim_z:
        :param dim_w:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param bottleneck:
        :param skip:
        :param norm_mean:
        :param norm_std:
        """
        super(GaussianMixtureVariationalAutoEncoderBig, self).__init__()

        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.dim_c = dim_c
        self.skip = skip

        self.encode_down = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                                   kernel_size=kernel_size, norm=norm, act=act, name_prefix='ENC_')

        self.encode_up = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels, out_ch=channels[0],
                                 strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final='identity',
                                 bottleneck=bottleneck, skip=skip, add_final=False, name_prefix='ENC_')

        self.decode_down = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                                   kernel_size=kernel_size, norm=norm, act=act, name_prefix='DEC_')

        self.decode_up = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels, out_ch=out_ch,
                                 strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final='sigmoid',
                                 bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='DEC_')

        z_channels = channels[0]
        # Z
        self.z_mu_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z, strides=1,
                                     kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='')
        self.z_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z,
                                            strides=1, kernel_size=1, conv_only=True, bias=True, norm='batch',
                                            act='relu', name='')
        self.z_log_sigma_conv.apply(weights_init)

        # W
        self.w_mu_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_w, strides=1,
                                     kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='')
        self.w_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_w,
                                            strides=1, kernel_size=1, conv_only=True, bias=True, norm='batch',
                                            act='relu', name='')
        self.w_log_sigma_conv.apply(weights_init)

        # Z_WC
        self.z_wc_conv = Convolution(dimensions=dimensions, in_channels=dim_w, out_channels=channels[0], strides=1,
                                     kernel_size=1, conv_only=False, bias=True, norm='batch', act='relu', name='')
        self.z_wc_mu_conv = Convolution(dimensions=dimensions, in_channels=channels[0], out_channels=dim_z*dim_w,
                                        strides=1, kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu')
        self.z_wc_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=channels[0], out_channels=dim_z*dim_w,
                                               strides=1, kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu')
        self.z_wc_log_sigma_conv.apply(weights_init)

    def _latent_distribution(self, z_enc: torch):
        #  q(z|x)
        z_mean = self.z_mu_conv(z_enc)
        z_log_sigma = self.z_log_sigma_conv(z_enc)
        z = z_mean + torch.exp(0.5 * z_log_sigma) * generate_tensor(z_mean)

        # q(w|x)
        w_mean = self.w_mu_conv(z_enc)
        w_log_sigma = self.w_log_sigma_conv(z_enc)
        w = w_mean + torch.exp(0.5 * w_log_sigma) * generate_tensor(w_mean)

        # posterior p(z|w,c)
        w_enc = self.z_wc_conv(w)
        z_wc_mean = self.z_wc_mu_conv(w_enc)
        z_wc_log_sigma = self.z_wc_log_sigma_conv(w_enc)
        z_wc = z_wc_mean + torch.exp(z_wc_log_sigma) * generate_tensor(z_wc_mean)

        # prior p(c) network
        z_sample = torch.tile(z, [1, self.dim_c, 1, 1])
        pc_logit = -0.5 * torch.pow(torch.subtract(z_sample, z_wc_mean), 2) * torch.exp(z_wc_log_sigma) \
                   - z_wc_log_sigma + torch.log(torch.tensor(np.pi))
        pc = self.softMax(pc_logit)

        return {'z': z, 'w': w, 'z_wc': z_wc, 'pc': pc}

    def _skip_forward(self, x: torch):
        # encoding network
        down_samples = []
        z_enc = x
        for layer in self.encode_down:
            z_enc = layer(z_enc)
            if hasattr(layer, 'conv'):
                down_samples.append(z_enc)
        for i, layer in enumerate(self.decode_up):
            if hasattr(layer, 'conv'):
                if layer.conv.stride[0] == 2:
                    z_enc = torch.cat((z_enc, down_samples[-i-1]), 1)
            z_enc = layer(z_enc)

        # latent
        return_dict = self._latent_distribution(z_enc)
        x_ = return_dict['z']

        # decoding network
        down_samples = []
        for layer in self.encode_down:
            down_samples.append(x_)
            x_ = layer(x_)
        for i, layer in enumerate(self.decode_up):
            x_ = layer(torch.cat((x_, down_samples[-i-1]), 1))
        return x_, return_dict

    def forward(self,  x: torch):
        if self.skip:
            return self._skip_forward(x)

        # encoding network
        z_enc = self.encode_up(self.encode_down(x))

        # latent
        return_dict = self._latent_distribution(z_enc)
        z = return_dict['z']

        # decoding network p(x|z) parameter
        x_ = self.decode_up(self.decode_down(z))
        return x_, return_dict
