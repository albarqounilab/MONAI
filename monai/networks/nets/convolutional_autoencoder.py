# Using nn.ModuleList
import torch.nn as nn
import numpy as np
import torch


class ConvAutoEncoder(nn.Module):
    """
    Simple auto-encoder implementation as in:
    Christoph Baur, Benedikt Wiestler, Shadi Albarqouni, and Nassir Navab. Deep Autoencoding Models for Unsupervised Anomaly Segmentation in Brain MR Images. arXiv preprint arXiv:1804.04488, 2018.
    """
    def __init__(self, input_shape=(1, 128, 128), intermediate_res=(16, 16), kernel_size=5, use_batchnorm=False,
                 filters_start=64, filters_max=128):
        """
        :param input_shape: tuple
            size of each dimension
        :param intermediate_res: int tuple
            desired intermediate resolution
        :param kernel_size: int
            kernel size of the convolution
        :param use_batchnorm: bool
            use batch normalization if True, else use group normalization
        :param filters_start: int
            amount of filter channels at first layer
        :param filters_max:
            amount of filter channels at bottleneck
        """
        super(ConvAutoEncoder, self).__init__()
        self.encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()
        self.intermediate_res = intermediate_res
        self.filters_start = filters_start
        self.filters_max = filters_max

        # ENCODER
        num_pooling = int(np.math.log(input_shape[1], 2) - np.math.log(float(intermediate_res[0]), 2))
        filters_in = 1
        for i in range(num_pooling):
            filters_out = int(min(self.filters_max, self.filters_start * (2 ** i)))
            self.encoding_layers.append(nn.Conv2d(filters_in, filters_out, kernel_size=kernel_size,stride=2, padding=2))
            self.encoding_layers.append(nn.BatchNorm2d(filters_out) if use_batchnorm else
                                        nn.GroupNorm(int(filters_out/2), filters_out))
            self.encoding_layers.append(nn.LeakyReLU())
            filters_in = filters_out

        # DECODER
        filters_in = self.filters_max
        self.decoding_layers.append(nn.BatchNorm2d(filters_in) if  use_batchnorm else
                                    nn.GroupNorm(int(filters_in/2), filters_in))
        self.decoding_layers.append(nn.ReLU())
        for i in range(num_pooling):
            filters_out = int(max(self.filters_start, self.filters_max / (2 ** i)))
            self.decoding_layers.append(nn.ConvTranspose2d(filters_in, filters_out,kernel_size,
                                                           stride=2, padding=2, output_padding=1))
            self.decoding_layers.append(nn.BatchNorm2d(filters_out) if use_batchnorm else
                                        nn.GroupNorm(int(filters_out/2), filters_out))
            self.decoding_layers.append(nn.LeakyReLU())
            filters_in = filters_out

        self.drop = nn.Dropout(0.2)
        self.merger = nn.Conv2d(self.filters_start, 1, 1, stride=1)

    def encode(self, x):
        z = x
        for i in range(len(self.encoding_layers)):
            z = self.encoding_layers[i](z)
        return z

    def decode(self, z):
        x_ = z
        for i in range(len(self.decoding_layers)):
            x_ = self.decoding_layers[i](x_)
        return x_

    def forward(self, x):
        # Encode
        z = self.encode(x)
        x_ = self.drop(z)

        # Decode
        x_ = self.decode(x_)
        # Merge
        x_ = torch.sigmoid(self.merger(x_))
        return x_, {'z':  z}
