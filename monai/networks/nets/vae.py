import torch
import torch.nn as nn
from typing import Sequence
from monai.networks.blocks import Convolution
from monai.networks.layers.convutils import weights_init
from monai.utils.misc import generate_tensor
from monai.networks.nets.convolutional_autoencoders import *

__all__ = ["VariationalAutoEncoder", "VariationalAutoEncoderBig"]


class VariationalAutoEncoder(nn.Module):

    def __init__(self, dimensions: int, in_channels: int, dim_z: int, channels: Sequence[int], out_ch: int,
                 strides: Sequence[int], kernel_size=5, norm='batch', act='leakyrelu', bottleneck=False, skip=False,
                 norm_mean=0, norm_std=1):
        """
        Variational Auto-Encoder
        :param dimensions:
        :param in_channels:
        :param dim_z:
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
        super(VariationalAutoEncoder, self).__init__()

        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.encode = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                              kernel_size=kernel_size, norm=norm, act=act, name_prefix='ENC_')

        self.decode = Decoder(dimensions=dimensions, in_channels=in_channels, channels=channels, out_ch=out_ch,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final='sigmoid',
                              bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='DEC_')

        z_channels = channels[-1]
        self.z_mu_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z, strides=1,
                                     kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='')

        self.z_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z, strides=1,
                                            kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='')
        self.z_log_sigma_conv.apply(weights_init)

    def forward(self,  x: torch):
        z_enc = self.encode(x)

        #  q(zS|x)
        z_mean = self.z_mu_conv(z_enc)
        z_log_sigma = self.z_log_sigma_conv(z_enc)

        z = z_mean + torch.exp(0.5 * z_log_sigma) * generate_tensor(z_mean, mean=self.norm_mean, sigma=self.norm_std)

        x_ = self.decode(z)
        return x_, {'z': z, 'z_mean': z_mean, 'z_log_sigma': z_log_sigma}


class VariationalAutoEncoderBig(nn.Module):

    def __init__(self, dimensions: int, in_channels: int, dim_z: int, channels: Sequence[int], out_ch: int,
                 strides: Sequence[int], kernel_size=5, norm='batch', act='leakyrelu', bottleneck=False, skip=False,
                 norm_mean=0, norm_std=1):
        """
        :param dimensions:
        :param in_channels:
        :param dim_z:
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
        super(VariationalAutoEncoderBig, self).__init__()

        self.norm_mean = norm_mean
        self.norm_std = norm_std
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
        self.z_mu_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z, strides=1,
                                     kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='')

        self.z_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z,
                                            strides=1,
                                            kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='')
        self.z_log_sigma_conv.apply(weights_init)

    def _skip_forward(self, x: torch):
        down_samples = []
        z_enc = x
        for layer in self.encode_down:
            z_enc = layer(z_enc)
            if hasattr(layer, 'conv'):
                down_samples.append(z_enc)
        for i, layer in enumerate(self.encode_up):
            print('shapes: {}- {} -- {}'.format(z_enc.shape, down_samples[-i - 1].shape, layer.conv.stride[0]))
            if hasattr(layer, 'conv'):
                if layer.conv.stride[0] == 2 and abs((-i-1)) <= len(down_samples):
                    z_enc = torch.cat((z_enc, down_samples[-i-1]), 1)
            z_enc = layer(z_enc)

        #  q(zS|x)
        z_mean = self.z_mu_conv(z_enc)
        z_log_sigma = self.z_log_sigma_conv(z_enc)

        z = z_mean + torch.exp(0.5 * z_log_sigma) * generate_tensor(z_mean, mean=self.norm_mean, sigma=self.norm_std)
        x_ = z
        down_samples = []
        for layer in self.decode_down:
            print('shape: {}'.format(x_.shape))
            x_ = layer(x_)
            if hasattr(layer, 'conv'):
                down_samples.append(x_)
        for i, layer in enumerate(self.decode_up):
            print('shape: {}'.format(x_.shape))
            if hasattr(layer, 'conv'):
                if layer.conv.stride[0] == 2 and abs((-i-1)) <= len(down_samples):
                    print('shapes: {}- {} -- {}'.format(z_enc.shape, down_samples[-i - 1].shape, layer.conv.stride[0]))
                    x_ = torch.cat((x_, down_samples[-i-1]), 1)
            x_ = layer(x_)
        return x_, {'z': z, 'z_mean': z_mean, 'z_log_sigma': z_log_sigma}

    def forward(self,  x: torch):
        if self.skip:
            return self._skip_forward(x)
        z_enc = self.encode_up(self.encode_down(x))

        #  q(zS|x)
        z_mean = self.z_mu_conv(z_enc)
        z_log_sigma = self.z_log_sigma_conv(z_enc)

        z = z_mean + torch.exp(0.5 * z_log_sigma) * generate_tensor(z_mean, mean=self.norm_mean, sigma=self.norm_std)

        x_ = self.decode_up(self.decode_down(z))
        return x_, {'z': z, 'z_mean': z_mean, 'z_log_sigma': z_log_sigma}
