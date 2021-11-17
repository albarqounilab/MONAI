import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from monai.networks.nets import AdaIN
from monai.networks.utils import *
import numpy as np


class StyleDecorator(torch.nn.Module):

    def __init__(self):
        super(StyleDecorator, self).__init__()

    def kernel_normalize(self, kernel, k=3):
        b, ch, h, w, kk = kernel.size()

        # calc kernel norm
        kernel = kernel.view(b, ch, h * w, kk).transpose(2, 1)
        kernel_norm = torch.norm(kernel.contiguous().view(b, h * w, ch * kk), p=2, dim=2, keepdim=True)

        # kernel reshape
        kernel = kernel.view(b, h * w, ch, k, k)
        kernel_norm = kernel_norm.view(b, h * w, 1, 1, 1)

        return kernel, kernel_norm

    def conv2d_with_style_kernels(self, features, kernels, patch_size, deconv_flag=False):
        output = list()
        b, c, h, w = features.size()

        # padding
        pad = (patch_size - 1) // 2
        padding_size = (pad, pad, pad, pad)

        # batch-wise convolutions with style kernels
        for feature, kernel in zip(features, kernels):
            # print(pad)
            feature = F.pad(feature.unsqueeze(0), padding_size, 'constant', 0)

            if deconv_flag:
                pad = patch_size - 1
                # padding_size = (pad, pad, pad, pad)
                output.append(F.conv_transpose2d(feature, kernel, padding=pad))
            else:
                output.append(F.conv2d(feature, kernel))

        return torch.cat(output, dim=0)

    def binarize_patch_score(self, features):
        outputs = list()

        # batch-wise operation
        for feature in features:
            matching_indices = torch.argmax(feature, dim=0)
            one_hot_mask = torch.zeros_like(feature)

            h, w = matching_indices.size()
            for i in range(h):
                for j in range(w):
                    ind = matching_indices[i, j]
                    one_hot_mask[ind, i, j] = 1
            outputs.append(one_hot_mask.unsqueeze(0))

        return torch.cat(outputs, dim=0)

    def norm_deconvolution(self, h, w, patch_size):
        mask = torch.ones((h, w))
        fullmask = torch.zeros((h + patch_size - 1, w + patch_size - 1))

        for i in range(patch_size):
            for j in range(patch_size):
                pad = (i, patch_size - i - 1, j, patch_size - j - 1)
                padded_mask = F.pad(mask, pad, 'constant', 0)
                fullmask += padded_mask

        pad_width = (patch_size - 1) // 2
        if pad_width == 0:
            deconv_norm = fullmask
        else:
            deconv_norm = fullmask[pad_width:-pad_width, pad_width:-pad_width]

        return deconv_norm.view(1, 1, h, w)

    def reassemble_feature(self, normalized_content_feature, normalized_style_feature, patch_size, patch_stride):
        # get patches of style feature
        style_kernel = extract_patches(normalized_style_feature, [patch_size, patch_size], [patch_stride, patch_stride])

        # kernel normalize
        style_kernel, kernel_norm = self.kernel_normalize(style_kernel, patch_size)

        # convolution with style kernel(patch wise convolution)
        patch_score = self.conv2d_with_style_kernels(normalized_content_feature, style_kernel / kernel_norm, patch_size)

        # binarization
        binarized = self.binarize_patch_score(patch_score)

        # deconv norm
        deconv_norm = self.norm_deconvolution(h=binarized.size(2), w=binarized.size(3), patch_size=patch_size)

        # deconvolution
        output = self.conv2d_with_style_kernels(binarized, style_kernel, patch_size, deconv_flag=True)

        return output / deconv_norm.type_as(output)

    def forward(self, content_feature, style_feature, style_strength=0.2, patch_size=3, patch_stride=1):
        # 1-1. content feature projection
        normalized_content_feature = whitening(content_feature)

        # 1-2. style feature projection
        normalized_style_feature = whitening(style_feature)

        # 2. swap content and style features
        reassembled_feature = self.reassemble_feature(normalized_content_feature, normalized_style_feature,
                                                      patch_size=patch_size, patch_stride=patch_stride)

        # 3. reconstruction feature with style mean and covariance matrix
        stylized_feature = coloring(reassembled_feature, style_feature)

        # 4. content and style interpolation
        result_feature = (1 - style_strength) * content_feature + style_strength * stylized_feature

        return result_feature


class ChamaeleoNet(nn.Module):
    """
    Disentangled AutoEncoder as in:
    Bercea, C. I., Wiestler, B., Rueckert, D., & Albarqouni, S. (2021). FedDis: Disentangled Federated Learning for Unsupervised Brain Pathology Segmentation. arXiv e-prints, arXiv-2103.
    """

    def __init__(self, input_shape=(1, 128, 128), intermediate_res=(16, 16), kernel_size=5, use_batchnorm=True,
                 filters_start_shape=64, filters_max_shape=128, filters_start_app=16, filters_max_app=32,
                 transformers=[AdaIN(), AdaIN(), AdaIN(), None]):

        super(ChamaeleoNet, self).__init__()
        # SHAPE
        self.shape_encoding_layers = nn.ModuleList()
        # APPEARANCE
        self.appearance_encoding_layers = nn.ModuleList()
        self.decoding_layers = nn.ModuleList()

        self.intermediate_res = intermediate_res
        self.filters_start_shape = filters_start_shape
        self.filters_max_shape = filters_max_shape
        self.filters_start_app = filters_start_app
        self.filters_max_app = filters_max_app
        self.filters_max = self.filters_max_shape + self.filters_max_app
        self.filters_start = self.filters_start_shape + self.filters_start_app

        self.adain = AdaIN()
        self.decorator = StyleDecorator()
        self.transformers = transformers

        # ENCODER SHAPE/APPEARANCE
        num_pooling = int(np.math.log(input_shape[1], 2) - np.math.log(float(intermediate_res[0]), 2))
        filters_in = 1
        filters_in_ = 1
        for i in range(num_pooling):
            filters_out = int(min(self.filters_max_shape, self.filters_start_shape * (2 ** i)))
            filters_out_ = int(min(self.filters_max_app, self.filters_start_app * (2 ** i)))

            # SHAPE
            self.shape_encoding_layers.append(
                nn.Conv2d(filters_in, filters_out, kernel_size=kernel_size, stride=2, padding=2))
            self.shape_encoding_layers.append(nn.BatchNorm2d(filters_out) if use_batchnorm else
                                              nn.GroupNorm(int(filters_out / 2), filters_out))
            self.shape_encoding_layers.append(nn.LeakyReLU())

            # APPEARANCE
            self.appearance_encoding_layers.append(
                nn.Conv2d(filters_in_, filters_out_, kernel_size=kernel_size, stride=2, padding=2))
            self.appearance_encoding_layers.append(nn.BatchNorm2d(filters_out_) if use_batchnorm else
                                                   nn.GroupNorm(int(filters_out_ / 2), filters_out_))
            self.appearance_encoding_layers.append(nn.LeakyReLU())
            filters_in = filters_out
            filters_in_ = filters_out_

        # DECODER
        filters_in = self.filters_max

        self.decoding_layers.append(nn.BatchNorm2d(filters_in) if use_batchnorm else
                                          nn.GroupNorm(int(filters_in / 2), filters_in))
        self.decoding_layers.append(nn.ReLU())

        for i in range(num_pooling):
            filters_out = int(max(self.filters_start, self.filters_max / (2 ** i)))

            self.decoding_layers.append(
                nn.ConvTranspose2d(filters_in, filters_out, kernel_size, stride=2, padding=2, output_padding=1))
            self.decoding_layers.append(
                nn.BatchNorm2d(filters_out) if use_batchnorm else nn.GroupNorm(int(filters_out / 2), filters_out))
            self.decoding_layers.append(nn.LeakyReLU())

            filters_in = filters_out

        self.decoder_drop = nn.Dropout(0.2)
        self.shape_drop = nn.Dropout(0.2)
        self.appearance_drop = nn.Dropout(0.2)

        self.merger = nn.Conv2d(self.filters_start, 1, 1, stride=1)

    def encode(self, x, path='shape'):
        z = x
        z_interm = []
        if path == 'shape':
            for i in range(len(self.shape_encoding_layers)):
                z = self.shape_encoding_layers[i](z)
        else:
            for i in range(len(self.appearance_encoding_layers)):
                # print('Shape z: {} with type{}'.format(z.shape, type(z)))
                z = self.appearance_encoding_layers[i](z)
                z_interm.append(z)
        return z, z_interm

    def decode(self, z, styles, masks=None, interpolation_weights=None):
        # x_ = self.decorator(z, styles[-1])  # z_CS
        x_ = torch.cat((z, styles[-1]), 1)
        for i in range(len(self.decoding_layers)):
            # print('Decode shape...{}'.format(x_.shape))
            x_ = self.decoding_layers[i](x_)  # Z_C
            # if self.transformers[i]:
            #     transformed_feature = []
            #     for style, interpolation_weight, mask in zip(styles, interpolation_weights, masks):
            #         if isinstance(mask, torch.Tensor):
            #             b, c, h, w = x_.size()
            #             mask = F.interpolate(mask, size=(h, w))
            #         transformed_feature.append(self.transformers[i](x_, style[i]) * interpolation_weight * mask)
            #     x_ = sum(transformed_feature)
        return x_

    def forward(self, content, style, style_strength=0.5, patch_size=3, patch_stride=1, masks=None,
                interpolation_weights=None, train=False):
        if interpolation_weights is None:
            interpolation_weights = [1 / len(style)] * len(style)
        if masks is None:
            masks = [1] * len(style)

        # encode content image
        content_feature, _ = self.encode(content, path='shape')
        zA, style_features = self.encode(style, path='appearance')

        # re-ordering style features for transferring feature during decoding
        # print('Shape size: {}, app size: {}'.format(content_feature.shape, zA.shape))
        # style_features = [torch.flip(style_feature[:-1], dims=[0, 1]) for style_feature in style_features]
        # style_features = [torch.flip(style_feature[:-1])[::-1] for style_feature in style_features]

        stylized_image = torch.sigmoid(self.merger(self.decode(content_feature, style_features, masks, interpolation_weights)))

        return stylized_image, {'zS': content_feature, 'zA': style_features}
