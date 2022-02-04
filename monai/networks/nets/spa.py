import torch
import torch.nn as nn
from typing import Sequence
from monai.networks.blocks import Convolution
from monai.networks.layers.convutils import weights_init
from monai.utils.misc import generate_tensor
from monai.networks.nets.convolutional_autoencoders import *
from torch.nn import Linear
import torch.nn.functional as F
__all__ = ["ShapePriorVAE"]


class ShapePriorVAE(nn.Module):

    def __init__(self, dimensions: int, in_channels: int, dim_z: int, dim_c: int, channels: Sequence[int], out_ch: int,
                 strides: Sequence[int], kernel_size=5, norm='batch', act='leakyrelu', bottleneck=False, skip=False,
                 norm_mean=0, norm_std=1, snr=0.0, add_appearance=False, channels_app=(0, 0, 0), strides_app=(0, 0, 0)):
        """
        :param dimensions:
        :param in_channels:
        :param dim_z:
        :param dim_c:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param bottleneck:
        :param skip:
        :param norm_mean:
        :param norm_std:
        :param snr:
        :param add_appearance:
        :param channels_app:
        :param strides_app:
        """
        super(ShapePriorVAE, self).__init__()

        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.snr = snr
        self.skip = skip
        self.add_appearance = add_appearance
        dim_appearance = 2  # TODO fix hard coded value
        decode_channels = dim_z * dim_c + dim_appearance if add_appearance else dim_z * dim_c

        self.encode_down = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                                   kernel_size=kernel_size, norm=norm, act=act, name_prefix='shape')

        self.encode_up = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels, out_ch=channels[0],
                                 strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final='identity',
                                 bottleneck=bottleneck, skip=skip, add_final=False, name_prefix='shape')

        self.decode_down = Encoder(dimensions=dimensions, in_channels=decode_channels, channels=channels, strides=strides,
                                   kernel_size=kernel_size, norm=norm, act=act, name_prefix='shape')

        self.decode_up = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels, out_ch=out_ch,
                                 strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final='sigmoid',
                                 bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='shape')

        z_channels = channels[0]
        self.z_mu_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z*dim_c, strides=1,
                                     kernel_size=1, conv_only=True, act='identity', name='shape_')

        self.z_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z*dim_c,
                                            strides=1, kernel_size=1, conv_only=True, act='identity', name='shape_')
        self.z_log_sigma_conv.apply(weights_init)

        if add_appearance:
            self.encode_app = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels_app,
                                       strides=strides_app, kernel_size=kernel_size, norm=norm, act=act, name_prefix='app')
            self.z_app_conv = Convolution(dimensions=dimensions, in_channels=channels_app[-1], out_channels=2,
                                         strides=1, kernel_size=1, conv_only=True, act='identity', name='app_')

        self.act_mean = torch.nn.Sigmoid()
        self.act_log_sigma = torch.nn.LogSigmoid()
        self.eps = 1e-8

    def _reparam(self, mu, sigma):
        eta = generate_tensor(mu, mean=self.norm_mean, sigma=self.norm_std)
        SNR = 1 if self.snr == 0.0 else self.snr * (torch.linalg.norm(mu) / torch.linalg.norm(eta))
        z = mu + torch.exp(0.5 * sigma) * SNR * eta
        return z

    def _latent_distribution(self, z_enc: torch):
        #  q(zS|x)
        z_mean = self.act_mean(self.z_mu_conv(z_enc))
        z_log_sigma = self.act_log_sigma(self.z_log_sigma_conv(z_enc) + self.eps)
        z = self._reparam(z_mean, z_log_sigma)
        return {'z': z, 'z_mean': z_mean, 'z_log_sigma': z_log_sigma}

    def _latent_appearance(self, z_enc_app: torch):
        zA = self.z_app_conv(z_enc_app)
        zA = zA.repeat(1, 1, 128, 128)
        return {'zA': zA}

    def _skip_helper(self, input, encoder, decoder):
        down_samples = []
        z_enc = input
        for layer in encoder:
            z_enc = layer(z_enc)
            if hasattr(layer, 'conv'):
                down_samples.append(z_enc)
        for i, layer in enumerate(decoder):
            if hasattr(layer, 'conv'):
                if layer.conv.stride[0] == 1 and abs((-i - 1)) <= len(down_samples):
                    z_enc = torch.cat((z_enc, down_samples[-i - 1]), 1)
            z_enc = layer(z_enc)
        return z_enc

    def _skip_forward(self, x: torch):
        # Encoder Shape
        z_enc = self._skip_helper(x, self.encode_down, self.encode_up)
        z_dict = self._latent_distribution(z_enc)
        x_ = z_dict['z']

        # Encoder Appearance
        if self.add_appearance:
            z_enc_app = self.encode_app(x)
            zA_dict = self._latent_appearance(z_enc_app)
            x_ = torch.cat((z_dict['z'], zA_dict['zA']), dim=1)
            z_dict.update(zA_dict)

        x_ = self._skip_helper(x_, self.decode_down, self.decode_up)
        return x_, z_dict

    def forward(self,  x: torch):
        if self.skip:
            return self._skip_forward(x)

        z_enc = self.encode_up(self.encode_down(x))

        z_dict = self._latent_distribution(z_enc)
        x_ = z_dict['z']

        if self.add_appearance:
            zA_dict = self._latent_appearance(x)
            x_ = self._concat(z_dict['z'], zA_dict['zA'])
            z_dict.update(zA_dict)

        x_ = self.decode_up(self.decode_down(x_))
        return x_, z_dict