import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from typing import Sequence
from monai.networks.blocks import Convolution
from monai.networks.layers.convutils import weights_init
from monai.utils.misc import generate_tensor
from monai.networks.nets.convolutional_autoencoders import *

__all__ = ["VariationalAutoEncoder", "VariationalAutoEncoderBig"]


class VariationalAutoEncoder(nn.Module):

    def __init__(self, dimensions: int, in_channels: int, dim_z: int, channels: Sequence[int], out_ch: int,
                 strides: Sequence[int], kernel_size=5, norm='batch', act='leakyrelu', bottleneck=False, skip=False,
                 norm_mean=0, norm_std=1, snr=0.0, add_appearance=False, channels_app=(0, 0, 0), strides_app=(0, 0, 0)):
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
        :param snr:
        """
        super(VariationalAutoEncoder, self).__init__()

        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.snr = snr
        self.add_appearance = add_appearance
        decode_channels = 2 * dim_z if add_appearance else dim_z

        self.encode = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                              kernel_size=kernel_size, norm=norm, act=act, name_prefix='shape')

        self.decode = Decoder(dimensions=dimensions, in_channels=decode_channels, channels=channels, out_ch=out_ch,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final='sigmoid',
                              bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='shape')

        z_channels = channels[-1]
        self.z_mu_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z, strides=1,
                                     kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='shape')

        self.z_log_sigma_conv = Convolution(dimensions=dimensions, in_channels=z_channels, out_channels=dim_z, strides=1,
                                            kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu', name='shape')
        self.z_log_sigma_conv.apply(weights_init)
        self.eps = 1e-8

        if add_appearance:
            self.encode_app = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels_app,
                                      strides=strides_app, kernel_size=kernel_size, norm=norm, act=act,
                                      name_prefix='app')
            self.fc1 = Linear(16 * 16 * channels_app[-1], 1024)
            self.fc2 = Linear(1024, 256)
            self.fc3 = Linear(256, 128)
            self.zA_mu_ = Linear(128, dim_z)
            self.zA_log_var_ = Linear(128, dim_z)
            self.zA_log_var_.apply(weights_init)

    def _reparam(self, mu, sigma):
        eta = generate_tensor(mu, mean=self.norm_mean, sigma=self.norm_std)
        SNR = 1 if self.snr == 0.0 else self.snr * (torch.linalg.norm(mu) / torch.linalg.norm(eta))
        z = mu + torch.exp(0.5 * sigma) * SNR * eta
        return z

    def _latent_distribution(self, z_enc: torch):
        #  q(zS|x)
        z_mean = self.z_mu_conv(z_enc)
        z_log_sigma = self.z_log_sigma_conv(z_enc) + self.eps
        z = self._reparam(z_mean, z_log_sigma)
        return {'z': z, 'z_mean': z_mean, 'z_log_sigma': z_log_sigma}

    def _latent_appearance(self, x: torch):
        z_enc_app = self.encode_app(x)
        z_enc_app = z_enc_app.view(z_enc_app.shape[0], -1)
        zA_enc = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(z_enc_app))))))
        zA_mean = self.zA_mu_(zA_enc)
        zA_log_sigma = self.zA_log_var_(zA_enc) + self.eps
        zA = self._reparam(zA_mean, zA_log_sigma)
        return {'zA': zA, 'zA_mean': zA_mean, 'zA_log_sigma': zA_log_sigma}

    def _concat(self, z:torch, zA:torch):
        b, c, h, w = z.shape
        zA = torch.unsqueeze(zA,-1)
        zA = torch.unsqueeze(zA,-1)
        zA = zA.repeat((1, 1, h, w))
        return torch.cat((z, zA), dim=1)

    def forward(self,  x: torch):
        z_enc = self.encode(x)

        #  q(zS|x)
        z_dict = self._latent_distribution(z_enc)
        z = z_dict['z']
        if self.add_appearance:
            zA_dict = self._latent_appearance(x)
            z = self._concat(z_dict['z'], zA_dict['zA'])
            z_dict.update(zA_dict)

        x_ = self.decode(z)
        return x_, z_dict


class VariationalAutoEncoderBig(nn.Module):

    def __init__(self, dimensions: int, in_channels: int, dim_z: int, channels: Sequence[int], out_ch: int,
                 strides: Sequence[int], kernel_size=5, norm='batch', act='leakyrelu', bottleneck=False, skip=False,
                 norm_mean=0, norm_std=1, snr=0.0, add_condition=False,
                 add_appearance=False, channels_app=(0, 0, 0), strides_app=(0, 0, 0)):
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
        :param snr:
        """
        super(VariationalAutoEncoderBig, self).__init__()

        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.skip = skip
        self.snr = snr
        self.add_appearance = add_appearance

        decode_channels = 2 * dim_z if add_appearance or add_condition else dim_z

        self.encode_down = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                                   kernel_size=kernel_size, norm=norm, act=act, name_prefix='ENC_')

        self.encode_up = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels, out_ch=channels[0],
                                 strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final='identity',
                                 bottleneck=bottleneck, skip=skip, add_final=False, name_prefix='ENC_')

        self.decode_down = Encoder(dimensions=dimensions, in_channels=decode_channels, channels=channels, strides=strides,
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
        self.eps = 1e-8

        if add_appearance:
            self.encode_app = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels_app,
                                      strides=strides_app, kernel_size=kernel_size, norm=norm, act=act,
                                      name_prefix='app')
            self.z_app_conv = Convolution(dimensions=dimensions, in_channels=channels_app[-1], out_channels=2,
                                          strides=1, kernel_size=1, conv_only=True, act='identity', name='app_')

    def _reparam(self, mu, sigma):
        eta = generate_tensor(mu, mean=self.norm_mean, sigma=self.norm_std)
        SNR = 1 if self.snr == 0.0 else self.snr * (torch.linalg.norm(mu) / torch.linalg.norm(eta))
        z = mu + torch.exp(0.5 * sigma) * SNR * eta
        return z

    def _latent_distribution(self, z_enc: torch):
        #  q(zS|x)
        z_mean = self.z_mu_conv(z_enc)
        z_log_sigma = self.z_log_sigma_conv(z_enc) + self.eps
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
        z = z_dict['z']
        # Encoder Appearance
        if self.add_appearance:
            z_enc_app = self.encode_app(x)
            zA_dict = self._latent_appearance(z_enc_app)
            x_ = torch.cat((z_dict['z'], zA_dict['zA']), dim=1)
            z_dict.update(zA_dict)

        x_ = self.decode_up(self.decode_down(z))
        return x_, z_dict
