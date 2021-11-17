from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F
from monai.networks.nets import Encoder


class PerceptualLoss(_Loss):
    """
    Wrapper for pytorch L1Loss
    see torch.nn.L1Loss(reduction='mean')

    """

    def __init__(
        self,
        reduction: str = 'mean',
        device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.reduction = reduction
        self.loss_network = Encoder().eval().to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        input_features = self.loss_network(input.repeat(1, 3, 1, 1))
        output_features = self.loss_network(target.repeat(1, 3, 1, 1))

        loss_pl = 0
        for output_feature, input_feature in zip(output_features, input_features):
            loss_pl += F.mse_loss(output_feature, input_feature)
        return loss_pl


class ContentLoss(_Loss):
    """
    Wrapper for pytorch L2Loss
    see torch.nn.L2Loss(reduction='mean')

    """

    def __init__(
            self,
            reduction: str = 'mean',
            device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.content_layers_default = ['conv_4']
        self.reduction = reduction
        self.loss_network = Encoder().eval().to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        input_features = self.loss_network(input.repeat(1, 3, 1, 1))
        output_features = self.loss_network(target.repeat(1, 3, 1, 1))
        # print('Len of vgg features is: {}'.format(len(output_features)))
        loss_pl = 0
        loss_pl += F.mse_loss(output_features[-1], input_features[-1])
        return loss_pl


class StyleLoss(_Loss):
    """
    Wrapper for pytorch L2Loss
    see torch.nn.L2Loss(reduction='mean')

    """

    def __init__(
            self,
            reduction: str = 'mean',
            device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.reduction = reduction
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        self.loss_network = Encoder().eval().to(self.device)

    @staticmethod
    def gram_matrix(input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        # print("Gram size is {} ". format(G.shape))
        return G.div(a * b * c * d)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        input_features = self.loss_network(input.repeat(1, 3, 1, 1))
        output_features = self.loss_network(target.repeat(1, 3, 1, 1))

        loss_pl = 0
        for input_feature, output_feature in zip(input_features, output_features):
            loss_pl += F.mse_loss(self.gram_matrix(input_feature), self.gram_matrix(output_feature))
        return loss_pl
