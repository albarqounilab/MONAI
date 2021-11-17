import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from monai.networks.nets import AdaIN
from monai.networks.utils import *


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
            feature = F.pad(feature.unsqueeze(0), padding_size, 'constant', 0)

            if deconv_flag:
                padding_size = patch_size - 1
                output.append(F.conv_transpose2d(feature, kernel, padding=padding_size))
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

    def forward(self, content_feature, style_feature, style_strength=1.0, patch_size=3, patch_stride=1):
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


class AvatarNet(nn.Module):
    def __init__(self, dim=2, layers=[1, 6, 11, 20]):
        super(AvatarNet, self).__init__()
        self.dim = dim
        self.encoder = Encoder(layers)
        self.decoder = Decoder(layers)

        self.adain = AdaIN()
        self.decorator = StyleDecorator()

    def forward(self, content, styles, style_strength=0.5, patch_size=3, patch_stride=1, masks=None,
                interpolation_weights=None, train=False):
        if interpolation_weights is None:
            interpolation_weights = [1 / len(styles)] * len(styles)
        if masks is None:
            masks = [1] * len(styles)

        # encode content image
        content_feature = self.encoder(content)
        style_features = []
        for style in styles:
            style_features.append(self.encoder(style))

        if not train:
            transformed_feature = []
            for style_feature, interpolation_weight, mask in zip(style_features, interpolation_weights, masks):
                if isinstance(mask, torch.Tensor):
                    b, c, h, w = content_feature[-1].size()
                    mask = F.interpolate(mask, size=(h, w))
                transformed_feature.append(
                    self.decorator(content_feature[-1], style_feature[-1], style_strength, patch_size,
                                   patch_stride) * interpolation_weight * mask)
            transformed_feature = sum(transformed_feature)

        else:
            transformed_feature = content_feature[-1]

        # re-ordering style features for transferring feature during decoding
        style_features = [style_feature[:-1][::-1] for style_feature in style_features]

        stylized_image = self.decoder(transformed_feature, style_features, masks, interpolation_weights)

        return stylized_image


class Encoder(nn.Module):
    def __init__(self, layers=[1, 6, 11, 20]):
        super(Encoder, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features

        self.encoder = nn.ModuleList()
        temp_seq = nn.Sequential()
        for i in range(max(layers) + 1):
            temp_seq.add_module(str(i), vgg[i])
            if i in layers:
                self.encoder.append(temp_seq)
                temp_seq = nn.Sequential()

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features


class Decoder(nn.Module):
    def __init__(self, layers=[1, 6, 11, 20], transformers=[AdaIN(), AdaIN(), AdaIN(), None]):
        super(Decoder, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=False).features
        self.transformers = transformers

        self.decoder = nn.ModuleList()
        temp_seq = nn.Sequential()
        count = 0
        for i in range(max(layers) - 1, -1, -1):
            if isinstance(vgg[i], nn.Conv2d):
                # get number of in/out channels
                out_channels = vgg[i].in_channels
                in_channels = vgg[i].out_channels
                kernel_size = vgg[i].kernel_size

                # make a [reflection pad + convolution + relu] layer
                temp_seq.add_module(str(count), nn.ReflectionPad2d(padding=(1, 1, 1, 1)))
                count += 1
                temp_seq.add_module(str(count), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size))
                count += 1
                temp_seq.add_module(str(count), nn.ReLU())
                count += 1

            # change down-sampling(MaxPooling) --> upsampling
            elif isinstance(vgg[i], nn.MaxPool2d):
                temp_seq.add_module(str(count), nn.Upsample(scale_factor=2))
                count += 1

            if i in layers:
                self.decoder.append(temp_seq)
                temp_seq = nn.Sequential()

        # append last conv layers without ReLU activation
        self.decoder.append(temp_seq[:-1])

    def forward(self, x, styles, masks=None, interpolation_weights=None):
        if interpolation_weights is None:
            interpolation_weights = [1 / len(styles)] * len(styles)
        if masks is None:
            masks = [1] * len(styles)

        y = x
        for i, layer in enumerate(self.decoder):
            y = layer(y)

            if self.transformers[i]:
                transformed_feature = []
                for style, interpolation_weight, mask in zip(styles, interpolation_weights, masks):
                    if isinstance(mask, torch.Tensor):
                        b, c, h, w = y.size()
                        mask = F.interpolate(mask, size=(h, w))
                    transformed_feature.append(self.transformers[i](y, style[i]) * interpolation_weight * mask)
                y = sum(transformed_feature)

        return y