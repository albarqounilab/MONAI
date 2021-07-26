import torch.nn as nn
import numpy as np
import torch


class DisAutoEncoder(nn.Module):
    """
    Disentangled AutoEncoder as in:
    Bercea, C. I., Wiestler, B., Rueckert, D., & Albarqouni, S. (2021). FedDis: Disentangled Federated Learning for Unsupervised Brain Pathology Segmentation. arXiv e-prints, arXiv-2103.
    """
    def __init__(self, input_shape=(1, 128, 128), intermediate_res=(16, 16), kernel_size=5, use_batchnorm=True,
                 is_dense=False, dense_size=128, filters_start_shape=64, filters_max_shape=128, filters_start_app=16,
                 filters_max_app=32):
        super(DisAutoEncoder, self).__init__()
        #SHAPE
        self.shape_encoding_layers = nn.ModuleList()
        self.shape_decoding_layers = nn.ModuleList()
        # APPEARANCE
        self.appearance_encoding_layers = nn.ModuleList()
        self.appearance_decoding_layers = nn.ModuleList()

        self.intermediate_res = intermediate_res
        self.is_dense = is_dense
        self.dense_size = dense_size
        self.nr_linear_neurons = 1024
        self.filters_start_shape = filters_start_shape
        self.filters_max_shape = filters_max_shape
        self.filters_start_app = filters_start_app
        self.filters_max_app = filters_max_app

        # ENCODER SHAPE/APPEARANCE
        num_pooling = int(np.math.log(input_shape[1], 2) - np.math.log(float(intermediate_res[0]), 2))
        filters_in = 1
        filters_in_ = 1
        for i in range(num_pooling):
            filters_out = int(min(self.filters_max_shape, self.filters_start_shape * (2 ** i)))
            filters_out_ = int(min(self.filters_max_app, self.filters_start_app * (2 ** i)))

            # SHAPE
            self.shape_encoding_layers.append(nn.Conv2d(filters_in, filters_out, kernel_size=kernel_size,stride=2, padding=2))
            self.shape_encoding_layers.append(nn.BatchNorm2d(filters_out) if use_batchnorm else
                                              nn.GroupNorm(int(filters_out/2), filters_out))
            self.shape_encoding_layers.append(nn.LeakyReLU())

            # APPEARANCE
            self.appearance_encoding_layers.append(nn.Conv2d(filters_in_, filters_out_, kernel_size=kernel_size, stride=2, padding=2))
            self.appearance_encoding_layers.append(nn.BatchNorm2d(filters_out_) if use_batchnorm else
                                                  nn.GroupNorm(int(filters_out_ / 2), filters_out_))
            self.appearance_encoding_layers.append(nn.LeakyReLU())
            filters_in = filters_out
            filters_in_ = filters_out_

        # BOTTLENECK
        if is_dense:
            filters_bottleneck = filters_out // 8
            self.shape_encoding_layers.append(nn.Conv2d(filters_out, filters_bottleneck, 1, stride=1))
            self.appearance_encoding_layers.append(nn.Conv2d(filters_out, filters_bottleneck, 1, stride=1))

            self.shape_encoding_layers.append(nn.Conv2d(filters_bottleneck, filters_out, 1, stride=1))
            self.appearance_decoding_layers.append(nn.Conv2d(filters_bottleneck, filters_out, 1, stride=1))

            self.shape_to_z = nn.Linear(self.nr_linear_neurons, dense_size)
            self.shape_from_z = nn.Linear(dense_size, self.nr_linear_neurons)

            self.appearance_to_z = nn.Linear(self.nr_linear_neurons, dense_size)
            self.appearance_from_z = nn.Linear(dense_size, self.nr_linear_neurons)


        # DECODER
        filters_in = self.filters_max_shape
        filters_in_ = self.filters_max_app

        self.shape_decoding_layers.append(nn.BatchNorm2d(filters_in) if use_batchnorm else
                                          nn.GroupNorm(int(filters_in/2), filters_in))
        self.shape_decoding_layers.append(nn.ReLU())

        self.appearance_decoding_layers.append(nn.BatchNorm2d(filters_in_) if use_batchnorm else
                                               nn.GroupNorm(int(filters_in_ / 2), filters_in_))
        self.appearance_decoding_layers.append(nn.ReLU())
        for i in range(num_pooling):
            filters_out = int(max(self.filters_start_shape, self.filters_max_shape / (2 ** i)))
            filters_out_ = int(max(self.filters_start_app, self.filters_max_app / (2 ** i)))

            self.shape_decoding_layers.append(nn.ConvTranspose2d(filters_in, filters_out, kernel_size, stride=2, padding=2, output_padding=1))
            self.shape_decoding_layers.append(nn.BatchNorm2d(filters_out) if use_batchnorm else nn.GroupNorm(int(filters_out/2), filters_out))
            self.shape_decoding_layers.append(nn.LeakyReLU())

            self.appearance_decoding_layers.append(nn.ConvTranspose2d(filters_in_, filters_out_,kernel_size, stride=2, padding=2, output_padding=1))
            self.appearance_decoding_layers.append(nn.BatchNorm2d(filters_out_) if use_batchnorm else nn.GroupNorm(int(filters_out_/2), filters_out_))
            self.appearance_decoding_layers.append(nn.LeakyReLU())
            filters_in = filters_out
            filters_in_ = filters_out_

        self.shape_drop = nn.Dropout(0.2)
        self.appearance_drop = nn.Dropout(0.2)

        self.merger = nn.Conv2d(self.filters_start_shape + self.filters_start_app, 1, 1, stride=1)

    def encode(self, x, path='shape'):
        z = x
        if path == 'shape':
            for i in range(len(self.shape_encoding_layers)):
                z = self.shape_encoding_layers[i](z)
        else:
            for i in range(len(self.appearance_encoding_layers)):
                z = self.appearance_encoding_layers[i](z)
        return z

    def decode(self, z, path='shape'):
        x_ = z
        if path == 'shape':
            for i in range(len(self.shape_decoding_layers)):
                x_ = self.shape_decoding_layers[i](x_)
        else:
            for i in range(len(self.appearance_decoding_layers)):
                x_ = self.appearance_decoding_layers[i](x_)
        return x_

    def forward(self, x):
        # Encode
        z_s = self.encode(x, path='shape')
        z_a = self.encode(x, path='appearance')
        x_s = self.shape_drop(z_s)
        x_a = self.appearance_drop(z_a)
        # Dense Bottleneck?
        if self.is_dense:
            x_s = x_s.view(-1,  self.nr_linear_neurons)
            z_s = self.to_z(x_s)
            x_s = self.from_z(z_s)
            x_s = x_s.view(-1,  16, self.intermediate_res[0], self.intermediate_res[1])
            x_a = x_a.view(-1, self.nr_linear_neurons)
            z_a = self.to_z(x_a)
            x_a = self.from_z(z_a)
            x_a = x_a.view(-1, 16, self.intermediate_res[0], self.intermediate_res[1])
        # Decode

        x_s = self.decode(x_s, path='shape')
        x_a = self.decode(x_a, path='appearance')
        # Merge
        # x_s_zeros = torch.zeros_like(x_s, device=x_s.device)
        # x_a_zeros = torch.zeros_like(x_a, device=x_a.device)

        x_ = torch.sigmoid(self.merger(torch.cat((x_s,  x_a), 1)))
        return {'x_rec': x_, 'z_s':  z_s, 'z_a':  z_a}