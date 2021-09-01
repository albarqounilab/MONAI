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

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__all__ = ["GaussianMixtureVariationalAutoEncoder"]


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
        dim_z: int = 1,
        dim_w:  int = 1,
        c_lambda: int = 1,
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

        # The number of channels and strides should match
        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        self.encoded_channels = in_channels
        decode_channel_list = list(channels[-2::-1]) + [out_channels]

        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)

        self.w_mu_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_w, 'q_wz_x/w_mean')
        self.w_log_sigma_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_w, 'q_wz_x/w_log_sigma')
        self.z_mu_conv, self.latent_channels = self._get_convolution_layer(self.encoded_channels, self.dim_z, 'q_wz_x/z_mean')
        self.z_log_sigma_conv, _ = self._get_convolution_layer(self.encoded_channels, self.dim_z, 'q_wz_x/z_log_sigma')
        self.z_wc_conv, _ = self._get_convolution_layer(self.dim_w, self.encoded_channels, 'p_z_wc/1x1convlayer', conv_only=False)
        self.z_wc_mu_conv, _ = self._get_convolution_layer(self.dim_w, self.dim_z*self.dim_c, 'p_z_wc/z_wc_mean')
        self.z_wc_log_sigma_conv, _ = self._get_convolution_layer(self.dim_w, self.dim_z*self.dim_c, 'p_z_wc/z_wc_log_sigma')

        self.decode, _ = self._get_decode_module(self.latent_channels, decode_channel_list, strides[::-1] or [1])



    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, False)
            encode.add_module("encode_%i" % i, layer)
            layer_channels = c

        return encode, layer_channels

    def _get_decode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        decode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_decode_layer(layer_channels, c, s, i == (len(strides) - 1))
            decode.add_module("decode_%i" % i, layer)
            layer_channels = c

        return decode, layer_channels

    def _get_encode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Module:

        if self.num_res_units > 0:
            return ResidualUnit(
                dimensions=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_last,
            )
        return Convolution(
            dimensions=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=is_last,
        )

    def _get_decode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Sequential:

        decode = nn.Sequential()

        conv = Convolution(
            dimensions=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            conv_only=is_last and self.num_res_units == 0,
            is_transposed=True,
        )

        decode.add_module("conv", conv)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                dimensions=self.dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_last,
            )

            decode.add_module("resunit", ru)

        return decode

    def _get_convolution_layer(self, dim_in, dim_out, name='xx', kernel_size=1, conv_only=True, act='prelu'):
        conv_layer = nn.Sequential()
        conv_layer.add_module(name,
                              Convolution(dimensions=self.dimensions, in_channels=dim_in, out_channels=dim_out,
                                          kernel_size=kernel_size, conv_only=conv_only, act=act))
        return conv_layer, dim_out

    @staticmethod
    def rand(input_var):
        rand_sample = torch.cuda.FloatTensor(input_var.shape).normal_() if input_var.is_cuda \
            else torch.FloatTensor(input_var.shape).normal_()
        return rand_sample

    def forward(self, x: torch.Tensor) -> Any:
        outputs = dict()

        #  Encoding network
        x = self.encode(x)

        #  q(z|x)
        outputs['z_mean'] = z_mean = self.z_mu_conv(x)
        outputs['z_log_sigma'] = z_log_sigma = self.z_log_sigma_conv(x)
        outputs['z_sampled'] = z_sampled = z_mean + torch.exp(0.5 * z_log_sigma) * self.rand(z_mean)

        # q(w|x)
        outputs['w_mean'] = w_mean = self.w_mu_conv(x)
        outputs['w_log_sigma'] = w_log_sigma = self.w_log_sigma_conv(x)
        outputs['w_sampled'] = w_sampled = w_mean + torch.exp(0.5 * w_log_sigma) * self.rand(w_mean)

        # posterior p(z|w,c)
        z_wc_mean = self.z_wc_mu_conv(w_sampled)
        z_wc_log_sigma = self.z_wc_log_sigma_conv(w_sampled)
        outputs['z_wc_mean'] = z_wc_means = torch.reshape(z_wc_mean, [-1, self.dim_z, z_wc_mean.shape[2], z_wc_mean.shape[3], self.dim_c])
        outputs['z_wc_log_sigma'] = z_wc_log_sigmas = torch.reshape(z_wc_log_sigma, [-1, self.dim_z, z_wc_log_sigma.shape[2], z_wc_log_sigma.shape[3],
                                                             self.dim_c])
        outputs['z_wc_sampled'] = z_wc_sampled = \
            z_wc_means + torch.exp(torch.tensor(-0.5 * z_wc_log_sigmas)) * self.rand(z_wc_means)

        # decoding network p(x|z) parameter
        outputs['x_rec'] = x_rec = self.decode(z_sampled)

        # prior p(c) network
        z_sample = torch.tile(torch.unsqueeze(z_sampled, -1), [1, 1, 1, 1, self.dim_c])
        pc_logit = -0.5 * torch.pow(torch.subtract(z_sample, z_wc_means), 2) * torch.exp(z_wc_log_sigmas)\
                   - z_wc_log_sigmas + torch.log(torch.tensor(np.pi))
        outputs['pc'] = pc = torch.softmax(torch.sum(pc_logit, 1), dim=-1)

        return outputs