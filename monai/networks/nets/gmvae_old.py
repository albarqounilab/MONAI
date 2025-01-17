# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import math

from monai.networks.blocks import Convolution, ResidualUnit, Upsample
from monai.networks.blocks import ADN
from monai.networks.layers.factories import Act, Norm
from monai.utils.misc import generate_tensor

__all__ = ["GaussianMixtureVariationalAutoEncoder", "GaussianMixtureVariationalAutoEncoder2",
           "DisentangledGaussianMixtureVariationalAutoEncoder"]


class GaussianMixtureVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        dim_c: int = 6,  # nr clusters
        dim_z: int = 128,
        dim_w:  int = 1,
        c_lambda: int = 0.5,
        restore_lr: float = 1e-3,
        restore_steps: int = 150,
        tv_lambda: float = 1.8
    ) -> None:

        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))
        self.dim_c = dim_c
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.c_lambda = c_lambda
        self.restore_lr = restore_lr
        self.restore_steps = restore_steps
        self.tv_lambda = tv_lambda
        self.act = 'relu'

        # The number of channels and strides should match
        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        self.encoded_channels = in_channels
        decode_channel_list = list(channels[-2::-1]) + [channels[0]] + [out_channels]

        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)

        self.z_mu_conv, self.latent_channels = self._get_convolution_layer(self.encoded_channels, self.dim_z,
                                                                           'q_wz_x/z_mean')
        self.z_log_sigma_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_z, 'q_wz_x/z_log_sigma')
        self.z_log_sigma_conv.apply(self.weights_init)

        self.w_mu_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_w, 'q_wz_x/w_mean')
        self.w_log_sigma_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_w, 'q_wz_x/w_log_sigma')
        self.w_log_sigma_conv.apply(self.weights_init)
        self.z_wc_conv, _ = self._get_convolution_layer(self.dim_w, 64, 'p_z_wc/1x1convlayer', conv_only=False)
        self.z_wc_mu_conv, _ = self._get_convolution_layer(64, self.dim_z*self.dim_c, 'p_z_wc/z_wc_mean')
        self.z_wc_log_sigma_conv, _ = self._get_convolution_layer(64, self.dim_z*self.dim_c, 'p_z_wc/z_wc_log_sigma', bias=False)
        self.z_wc_log_sigma_conv.apply(self.weights_init)

        self.decode, _ = self._get_decode_module(self.latent_channels, decode_channel_list, strides[::-1] or [1])
        self.softMax = torch.nn.Softmax()

    @staticmethod
    def weights_init(m):
        if hasattr(m, 'weight'):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    @staticmethod
    def _get_encode_module(in_channels: int, channels: Sequence[int], strides: Sequence[int]):
        layer_channels = in_channels
        encoder = nn.Sequential()
        for i, (c, s) in enumerate(zip(channels, strides)):
            encoder.add_module("encode_%i" %i, Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu'))
            layer_channels = c
        return encoder, layer_channels

    @staticmethod
    def _get_decode_module(in_channels: int, channels: Sequence[int], strides: Sequence[int]):
        decoder = nn.Sequential()
        layer_channels = in_channels
        decoder.add_module('decode_init', ADN(in_channels=in_channels, act='leakyrelu', norm='batch', norm_dim=2))
        for i, (c, s) in enumerate(zip(channels, strides)):
            decoder.add_module("decode_%i" %i, Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu',
                                                           is_transposed=True))
            layer_channels = c

        out_ch = channels[-1]
        decoder.add_module("decode_final", Convolution(dimensions=2, in_channels=layer_channels, out_channels=out_ch,
                                                       strides=1, kernel_size=1, conv_only=True, act='identity'))
        return decoder, layer_channels

    def _get_convolution_layer(self, dim_in, dim_out, name='', kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu'):
        conv_layer = nn.Sequential()
        conv_layer.add_module(name,
                              Convolution(dimensions=self.dimensions, in_channels=dim_in, out_channels=dim_out,
                                          kernel_size=kernel_size, conv_only=conv_only, bias=bias, norm=norm, act=act))
        return conv_layer, dim_out

    def forward(self, x: torch.Tensor, scanner: torch.Tensor) -> Any:
        outputs = dict()

        #  Encoding network
        z_enc = self.encode(x)

        #  q(z|x)
        outputs['z_mean'] = z_mean = self.z_mu_conv(z_enc)
        outputs['z_log_sigma'] = z_log_sigma = self.z_log_sigma_conv(z_enc)
        outputs['z_sampled'] = z_sampled = z_mean + torch.exp(0.5 * z_log_sigma) * generate_tensor(z_mean)

        # q(w|x)
        outputs['w_mean'] = w_mean = self.w_mu_conv(z_enc)
        outputs['w_log_sigma'] = w_log_sigma = self.w_log_sigma_conv(z_enc)
        outputs['w_sampled'] = w_sampled = w_mean + torch.exp(0.5 * w_log_sigma) * generate_tensor(w_mean)

        # posterior p(z|w,c)
        w_sampled = self.z_wc_conv(w_sampled)
        outputs['z_wc_mean'] = z_wc_means = self.z_wc_mu_conv(w_sampled)
        z_wc_log_sigma = self.z_wc_log_sigma_conv(w_sampled)
        outputs['z_wc_log_sigma'] = z_wc_log_sigmas = z_wc_log_sigma + generate_tensor(z_wc_log_sigma, 0.1)  # Add 0.1 bias
        outputs['z_wc_sampled'] = z_wc_log_sigmas = \
            z_wc_means + torch.exp(z_wc_log_sigmas) * generate_tensor(z_wc_means)

        # decoding network p(x|z) parameter
        outputs['x_mean'] = self.decode(z_sampled)

        # prior p(c) network
        z_sample = torch.tile(z_sampled, [1, self.dim_c, 1, 1])
        pc_logit = -0.5 * torch.pow(torch.subtract(z_sample, z_wc_means), 2) * torch.exp(z_wc_log_sigmas)\
                   - z_wc_log_sigmas + torch.log(torch.tensor(np.pi))
        outputs['pc'] = self.softMax(pc_logit)

        return outputs


class GaussianMixtureVariationalAutoEncoder2(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        dim_c: int = 6,  # nr clusters
        dim_z: int = 128,
        dim_w:  int = 1,
        c_lambda: int = 0.5,
        restore_lr: float = 1e-3,
        restore_steps: int = 150,
        tv_lambda: float = 1.8
    ) -> None:

        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))
        self.dim_c = dim_c
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.c_lambda = c_lambda
        self.restore_lr = restore_lr
        self.restore_steps = restore_steps
        self.tv_lambda = tv_lambda
        self.act = 'relu'

        # The number of channels and strides should match
        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        self.encoded_channels = in_channels
        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)

        self.z_mu_conv, self.latent_channels = self._get_convolution_layer(self.encoded_channels, self.dim_z,
                                                                           'q_wz_x/z_mean')
        self.z_log_sigma_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_z, 'q_wz_x/z_log_sigma')
        self.z_log_sigma_conv.apply(self.weights_init)

        self.w_mu_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_w, 'q_wz_x/w_mean')
        self.w_log_sigma_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_w, 'q_wz_x/w_log_sigma')
        self.w_log_sigma_conv.apply(self.weights_init)
        self.z_wc_conv, _ = self._get_convolution_layer(self.dim_w, 64, 'p_z_wc/1x1convlayer', conv_only=False)
        self.z_wc_mu_conv, _ = self._get_convolution_layer(64, self.dim_z*self.dim_c, 'p_z_wc/z_wc_mean')
        self.z_wc_log_sigma_conv, _ = self._get_convolution_layer(64, self.dim_z*self.dim_c, 'p_z_wc/z_wc_log_sigma', bias=False)
        self.z_wc_log_sigma_conv.apply(self.weights_init)

        self.decode, _ = self._get_decode_module(self.latent_channels, channels, strides, out_channels)
        self.softMax = torch.nn.Softmax()

    @staticmethod
    def weights_init(m):
        if hasattr(m, 'weight'):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    @staticmethod
    def _get_encode_module(in_channels: int, channels: Sequence[int], strides: Sequence[int]):
        decode_channel_list = list(channels[-1::-1]) + [channels[0]]
        decode_strides = strides[::-1] or [1]
        layer_channels = in_channels
        encoder = nn.Sequential()
        for i, (c, s) in enumerate(zip(channels, strides)):
            encoder.add_module("encode_down_%i" %i, Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu'))
            layer_channels = c
        for i, (c, s) in enumerate(zip(decode_channel_list, decode_strides)):
            encoder.add_module("encode_up_%i" % i,
                               Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu',is_transposed=True))
            layer_channels = c
        return encoder, layer_channels

    @staticmethod
    def _get_decode_module(in_channels: int, channels: Sequence[int], strides: Sequence[int], out_ch):
        decode_channel_list = list(channels[-2::-1]) + [channels[0]]
        decode_strides = strides[::-1] or [1]
        layer_channels = in_channels
        decoder = nn.Sequential()
        for i, (c, s) in enumerate(zip(channels, strides)):
            decoder.add_module("decode_down_%i" % i,
                               Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu'))
            layer_channels = c
        for i, (c, s) in enumerate(zip(decode_channel_list, decode_strides)):
            decoder.add_module("decode_up_%i" % i,
                               Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu', is_transposed=True))
            layer_channels = c

        decoder.add_module("decode_final", Convolution(dimensions=2, in_channels=layer_channels, out_channels=out_ch,
                                                       strides=1, kernel_size=1, conv_only=True, act='identity'))
        return decoder, layer_channels

    def _get_convolution_layer(self, dim_in, dim_out, name='', kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu'):
        conv_layer = nn.Sequential()
        conv_layer.add_module(name,
                              Convolution(dimensions=self.dimensions, in_channels=dim_in, out_channels=dim_out,
                                          kernel_size=kernel_size, conv_only=conv_only, bias=bias, norm=norm, act=act))
        return conv_layer, dim_out

    def forward(self, x: torch.Tensor, scanner: torch.Tensor) -> Any:
        outputs = dict()

        #  Encoding network
        z_enc = self.encode(x)

        #  q(z|x)
        outputs['z_mean'] = z_mean = self.z_mu_conv(z_enc)
        outputs['z_log_sigma'] = z_log_sigma = self.z_log_sigma_conv(z_enc)
        outputs['z_sampled'] = z_sampled = z_mean + torch.exp(0.5 * z_log_sigma) * generate_tensor(z_mean)

        # q(w|x)
        outputs['w_mean'] = w_mean = self.w_mu_conv(z_enc)
        outputs['w_log_sigma'] = w_log_sigma = self.w_log_sigma_conv(z_enc)
        outputs['w_sampled'] = w_sampled = w_mean + torch.exp(0.5 * w_log_sigma) * generate_tensor(w_mean)

        # posterior p(z|w,c)
        w_sampled = self.z_wc_conv(w_sampled)
        outputs['z_wc_mean'] = z_wc_means = self.z_wc_mu_conv(w_sampled)
        z_wc_log_sigma = self.z_wc_log_sigma_conv(w_sampled)
        outputs['z_wc_log_sigma'] = z_wc_log_sigmas = z_wc_log_sigma +generate_tensor(z_wc_log_sigma, 0.1)  # Add 0.1 bias
        outputs['z_wc_sampled'] = z_wc_log_sigmas = \
            z_wc_means + torch.exp(z_wc_log_sigmas) * generate_tensor(z_wc_means)

        # decoding network p(x|z) parameter
        outputs['x_mean'] = self.decode(z_sampled)

        # prior p(c) network
        z_sample = torch.tile(z_sampled, [1, self.dim_c, 1, 1])
        pc_logit = -0.5 * torch.pow(torch.subtract(z_sample, z_wc_means), 2) * torch.exp(z_wc_log_sigmas)\
                   - z_wc_log_sigmas + torch.log(torch.tensor(np.pi))
        outputs['pc'] = self.softMax(pc_logit)

        return outputs


class DisentangledGaussianMixtureVariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: Optional[Union[Tuple, str, float]] = None,
        dim_c: int = 6,  # nr clusters
        dim_z: int = 128,
        dim_w:  int = 1,
        c_lambda: int = 0.5,
        restore_lr: float = 1e-3,
        restore_steps: int = 150,
        tv_lambda: float = 1.8
    ) -> None:

        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))
        self.dim_c = dim_c
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.c_lambda = c_lambda
        self.restore_lr = restore_lr
        self.restore_steps = restore_steps
        self.tv_lambda = tv_lambda
        self.act = 'relu'

        # The number of channels and strides should match
        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        self.encoded_channels = in_channels
        channels_appearance = [int(channel / 4) for channel in channels]
        self.encode_shape, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)
        self.encode_appearance, self.encoded_app_channels = \
            self._get_encode_module(in_channels + 1, channels_appearance,  strides, name='_appearance')

        self.z_mu_conv, self.latent_channels = self._get_convolution_layer(self.encoded_channels, self.dim_z,
                                                                           'q_wz_x/z_mean')
        self.z_log_sigma_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_z, 'q_wz_x/z_log_sigma')
        self.z_log_sigma_conv.apply(self.weights_init)

        self.zA_mu_conv, _ = self._get_convolution_layer(self.encoded_app_channels, self.dim_z, 'q_wz_x/zA_mean')
        self.zA_log_sigma_conv, _ = self._get_convolution_layer(self.encoded_app_channels, self.dim_z, 'q_wz_x/zA_log_sigma')
        self.zA_log_sigma_conv.apply(self.weights_init)

        self.w_mu_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_w, 'q_wz_x/w_mean')
        self.w_log_sigma_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_w, 'q_wz_x/w_log_sigma')
        self.w_log_sigma_conv.apply(self.weights_init)
        self.z_wc_conv, _ = self._get_convolution_layer(self.dim_w, 64, 'p_z_wc/1x1convlayer', conv_only=False)
        self.z_wc_mu_conv, _ = self._get_convolution_layer(64, self.dim_z*self.dim_c, 'p_z_wc/z_wc_mean')
        self.z_wc_log_sigma_conv, _ = self._get_convolution_layer(64, self.dim_z*self.dim_c, 'p_z_wc/z_wc_log_sigma', bias=False)
        self.z_wc_log_sigma_conv.apply(self.weights_init)

        self.decode, _ = self._get_decode_module(self.latent_channels * 2 + 1, channels, strides, out_channels)
        self.softMax = torch.nn.Softmax(dim=0)

    @staticmethod
    def weights_init(m):
        if hasattr(m, 'weight'):
            torch.nn.init.zeros_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    @staticmethod
    def _get_encode_module(in_channels: int, channels: Sequence[int], strides: Sequence[int], name='_shape'):
        decode_channel_list = list(channels[-1::-1])
        decode_strides = strides[::-1] or [1]
        layer_channels = in_channels
        encoder = nn.Sequential()
        for i, (c, s) in enumerate(zip(channels, strides)):
            encoder.add_module("encode_down_" + name + "%i" %i, Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu'))
            layer_channels = c
        for i, (c, s) in enumerate(zip(decode_channel_list, decode_strides)):
            encoder.add_module("encode_up_" + name + "%i" % i,
                               Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu',is_transposed=True))
            layer_channels = c
        return encoder, layer_channels

    @staticmethod
    def _get_decode_module(in_channels: int, channels: Sequence[int], strides: Sequence[int], out_ch):
        decode_channel_list = list(channels[-1::-1])
        decode_strides = strides[::-1] or [1]
        layer_channels = in_channels
        decoder = nn.Sequential()
        for i, (c, s) in enumerate(zip(channels, strides)):
            decoder.add_module("decode_down_%i" % i,
                               Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu'))
            layer_channels = c
        for i, (c, s) in enumerate(zip(decode_channel_list, decode_strides)):
            decoder.add_module("decode_up_%i" % i,
                               Convolution(dimensions=2, in_channels=layer_channels, out_channels=c,
                                           strides=s, kernel_size=5, norm='batch', act='leakyrelu', is_transposed=True))
            layer_channels = c

        decoder.add_module("decode_final", Convolution(dimensions=2, in_channels=layer_channels, out_channels=out_ch,
                                                       strides=1, kernel_size=1, conv_only=True, act='identity'))
        return decoder, layer_channels

    def _get_convolution_layer(self, dim_in, dim_out, name='', kernel_size=1, conv_only=True, bias=True, norm='batch', act='relu'):
        conv_layer = nn.Sequential()
        conv_layer.add_module(name,
                              Convolution(dimensions=self.dimensions, in_channels=dim_in, out_channels=dim_out,
                                          kernel_size=kernel_size, conv_only=conv_only, bias=bias, norm=norm, act=act))
        return conv_layer, dim_out

    def forward(self, x: torch.Tensor, scanner: torch.Tensor) -> Any:
        outputs = dict()

        #  Encoding network
        z_enc = self.encode_shape(x)
        zA_enc = self.encode_appearance(torch.cat([x, scanner], dim=1))

        #  q(z|x)
        outputs['z_mean'] = z_mean = self.z_mu_conv(z_enc)
        outputs['z_log_sigma'] = z_log_sigma = self.z_log_sigma_conv(z_enc)
        outputs['z_sampled'] = z_sampled = z_mean + torch.exp(0.5 * z_log_sigma) * generate_tensor(z_mean)

        #  q(zA|x)
        outputs['zA_mean'] = zA_mean = self.zA_mu_conv(zA_enc)
        outputs['zA_log_sigma'] = zA_log_sigma = self.zA_log_sigma_conv(zA_enc)
        outputs['zA_sampled'] = zA_sampled = zA_mean + torch.exp(0.5 * zA_log_sigma) * generate_tensor(zA_mean)

        # q(w|x)
        outputs['w_mean'] = w_mean = self.w_mu_conv(z_enc)
        outputs['w_log_sigma'] = w_log_sigma = self.w_log_sigma_conv(z_enc)
        outputs['w_sampled'] = w_sampled = w_mean + torch.exp(0.5 * w_log_sigma) * generate_tensor(w_mean)

        # posterior p(z|w,c)
        w_sampled = self.z_wc_conv(w_sampled)
        outputs['z_wc_mean'] = z_wc_means = self.z_wc_mu_conv(w_sampled)
        z_wc_log_sigma = self.z_wc_log_sigma_conv(w_sampled)
        outputs['z_wc_log_sigma'] = z_wc_log_sigmas = z_wc_log_sigma + generate_tensor(z_wc_log_sigma, 0.1)  # Add 0.1 bias
        outputs['z_wc_sampled'] = z_wc_log_sigmas = \
            z_wc_means + torch.exp(z_wc_log_sigmas) * generate_tensor(z_wc_means)

        # decoding network p(x|z) parameter
        outputs['x_mean'] = self.decode(torch.cat([z_sampled, zA_sampled, scanner], dim=1))

        # prior p(c) network
        z_sample = torch.tile(z_sampled, [1, self.dim_c, 1, 1])
        pc_logit = -0.5 * torch.pow(torch.subtract(z_sample, z_wc_means), 2) * torch.exp(z_wc_log_sigmas)\
                   - z_wc_log_sigmas + torch.log(torch.tensor(np.pi))
        outputs['pc'] = self.softMax(pc_logit)

        return outputs