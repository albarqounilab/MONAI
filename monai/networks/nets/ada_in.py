import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def forward(self, content, style, style_strength=1.0, eps=1e-5):
        b, c, h, w = content.size()

        content_std, content_mean = torch.std_mean(content.view(b, c, -1), dim=2, keepdim=True)
        style_std, style_mean = torch.std_mean(style.view(b, c, -1), dim=2, keepdim=True)

        normalized_content = (content.view(b, c, -1) - content_mean) / (content_std + eps)

        stylized_content = (normalized_content * style_std) + style_mean

        output = (1 - style_strength) * content + style_strength * stylized_content.view(b, c, h, w)
        return output