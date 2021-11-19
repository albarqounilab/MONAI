import torch
import torch.nn as nn
import numpy as np
from typing import Sequence
from monai.networks.blocks import Convolution

__all__ = ["Encoder", "Decoder", 'ConvAutoEncoder', 'DisAutoEncoder']


class Encoder(nn.Sequential):
    def __init__(self, dimensions: int, in_channels: int, channels: Sequence[int], strides: Sequence[int],
                 kernel_size=5, norm='batch', act='leakyrelu', name_prefix='shape'):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param name_prefix:
        """
        super(Encoder, self).__init__()

        layer_channels = in_channels
        for i, (c, s) in enumerate(zip(channels, strides)):
            self.add_module(name_prefix + "_encode_%i" % i,
                            Convolution(dimensions=dimensions, in_channels=layer_channels, out_channels=c,
                                        strides=s, kernel_size=kernel_size, norm=norm, act=act))
            layer_channels = c


class Decoder(nn.Sequential):
    def __init__(self, dimensions: int, in_channels: int, channels: Sequence[int], out_ch: int, strides: Sequence[int],
                 kernel_size: int = 5, norm: str = 'batch', act: str = 'leakyrelu', act_final: str = 'sigmoid',
                 bottleneck: bool = False, skip: bool = False, add_final: bool = True, name_prefix: str = '_'):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        :param add_final:
        :param name_prefix:
        """
        super(Decoder, self).__init__()

        decode_channel_list = list(channels[-1::-1])
        lr, nr_layers = 0, strides.count(2)
        decode_strides = strides #strides[::-1] or [1]
        layer_channels = in_channels
        for i, (c, s) in enumerate(zip(decode_channel_list, decode_strides)):
            if i == 0 and bottleneck:
                continue
            if skip and s == 2 and lr < nr_layers:
                lr += 1
                layer_channels = layer_channels + channels[-i-1]
            self.add_module(name_prefix + "_decode_%i" % i,
                            Convolution(dimensions=dimensions, in_channels=layer_channels, out_channels=c,
                                        strides=s, kernel_size=kernel_size, norm=norm, act=act, is_transposed=True))
            layer_channels = c

        if add_final:
            self.add_module(name_prefix + "_decode_final",
                            Convolution(dimensions=dimensions, in_channels=layer_channels, out_channels=out_ch,
                                        strides=1, kernel_size=1, conv_only=True, act=act_final))


class ConvAutoEncoder(nn.Module):

    def __init__(self, dimensions: int, in_channels: int, channels: Sequence[int], out_ch: int, strides: Sequence[int],
                 kernel_size=5, norm='batch', act='leakyrelu', act_final='sigmoid', bottleneck=False, skip=False):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        """
        self.encode = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                              kernel_size=kernel_size, norm=norm, act=act, name_prefix='conv_')

        self.decode = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels, out_ch=out_ch,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final=act_final,
                              bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='conv_')
        super(ConvAutoEncoder, self).__init__()

    def forward(self,  x: torch):
        z = self.encode(x)
        x_ = self.decode(z)
        return x_, {'z': z}


class DisAutoEncoder(nn.Module):

    def __init__(self, dimensions: int, in_channels: int, channels: Sequence[int], out_ch: int, strides: Sequence[int],
                 channels_app: Sequence[int], kernel_size=5, norm='batch', act='leakyrelu', act_final='sigmoid',
                 bottleneck=False, skip=False):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param out_ch:
        :param strides:
        :param channels_app:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        """
        super(DisAutoEncoder, self).__init__()

        self.encode_shape = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                                    kernel_size=kernel_size, norm=norm, act=act, name_prefix='shape_')
        self.encode_app = Encoder(dimensions=dimensions, in_channels=channels[-1], channels=channels_app,
                                  strides=strides, kernel_size=kernel_size, norm=norm, act=act, name_prefix='app_')

        channels_decoder = [a + b for a, b in zip(channels, channels_app)]
        self.decode = Decoder(dimensions=dimensions, in_channels=channels_decoder[-1], channels=channels_decoder,
                              out_ch=out_ch, strides=strides, kernel_size=kernel_size, norm=norm, act=act,
                              act_final=act_final, bottleneck=bottleneck, skip=skip, add_final=True, name_prefix='shape_')

    def forward(self,  x: torch):
        zS = self.encode_shape(x)
        zA = self.encode_app(x)
        z = torch.cat((zS, zA), 1)
        x_ = self.decode(z)
        return x_, {'z': z, 'zS': zS, 'zA': zA}
