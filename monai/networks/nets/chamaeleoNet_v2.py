import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, num_block=1, norm='none', n_group=32, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        curr_dim = dim
        layers += [LinearBlock(input_dim, curr_dim, norm=norm, n_group=n_group, activation=activation)]

        for _ in range(num_block):
            layers += [LinearBlock(curr_dim, curr_dim, norm=norm, n_group=n_group, activation=activation)]

        layers += [LinearBlock(curr_dim, output_dim, norm='none', activation='none')]  # no output activations
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x.view(x.size(0), -1))


class Get(object):
    def __init__(self, s_CT, C, G, mask):
        self.s_CT = s_CT
        self.C = C
        self.G = G
        self.mask = mask
        self.n_mem = C // G

    def coloring(self):
        X = []  # coloring matrix
        U_arr = []

        for i in range(self.G):  # This is the same with 'i' in Fig.3(b) in the paper.
            # B,n_mem,n_mem
            s_CT_i = self.s_CT[:, (self.n_mem ** 2) * i: (self.n_mem ** 2) * (i + 1)].unsqueeze(2).view(
                self.s_CT.size(0), self.n_mem, self.n_mem)
            D = (torch.sum(s_CT_i ** 2, dim=1,
                           keepdim=True)) ** 0.5  # Compute the comlumn-wise L2 norm of s_CT_i (we assume that D is the eigenvalues) / B,n_mem,n_mem => B,1,n_mem
            U_i = s_CT_i / D  # B,n_mem,n_mem
            UDU_T_i = torch.bmm(s_CT_i, U_i.permute(0, 2, 1))  # B,n_mem,n_mem

            X += [UDU_T_i]
            U_arr += [U_i]

        eigen_s = torch.cat(U_arr, dim=0)  # eigen_s is used in the coloring regularization / B*G,n_mem,n_mem
        X = torch.cat(X, dim=1)  # B,G*n_mem,n_mem
        X = X.repeat(1, 1, self.G)  # B,C,C
        X = self.mask * X

        return X, eigen_s


class WCT(nn.Module):
    def __init__(self, n_group, device, input_dim, mlp_dim, bias_dim, mask, w_alpha=0.4):
        super(WCT, self).__init__()
        self.G = n_group
        self.device = device
        self.alpha = nn.Parameter(torch.ones(1) - w_alpha)
        self.mlp_CT = MLP(input_dim // n_group, (input_dim // n_group) ** 2, dim=mlp_dim, num_block=3, norm='none',
                          n_group=n_group, activation='lrelu')
        self.mlp_mu = MLP(input_dim, bias_dim, dim=input_dim, num_block=1, norm='none', n_group=n_group,
                          activation='lrelu')
        self.mask = mask

    def forward(self, c_A, s_B):
        return self.wct(c_A, s_B)

    def wct(self, c_A, s_B):
        '''
        style_size torch.Size([1, 766])
        mask_size torch.Size([1, 1, 64, 64])
        content_size torch.Size([1, 256, 64, 64])
        W_size torch.size([1,256,256])
        '''

        B, C, H, W = c_A.size()
        n_mem = C // self.G  # 32 if G==8

        s_B_CT = self.mlp_CT(s_B.view(B * self.G, C // self.G, 1, 1)).view(B, -1)  # B*G,C//G,1,1 => B,G*(C//G)**2
        s_B_mu = self.mlp_mu(s_B).unsqueeze(2).unsqueeze(3)

        X_B, eigen_s = Get(s_B_CT, c_A.size(1), self.G, self.mask).coloring()

        eps = 1e-5
        c_A_ = c_A.permute(1, 0, 2, 3).contiguous().view(self.G, n_mem, -1)  # B,C,H,W => C,B,H,W => G,C//G,BHW
        c_A_mean = torch.mean(c_A_, dim=2, keepdim=True)
        c_A_ = c_A_ - c_A_mean  # G,C//G,BHW

        cov_c = torch.bmm(c_A_, c_A_.permute(0, 2, 1)).div(B * H * W - 1) + eps * torch.eye(n_mem).unsqueeze(0).to(
            self.device)  # G,C//G,C//G

        whitend = c_A_.unsqueeze(0).contiguous().view(C, B, -1).permute(1, 0, 2)  # B,C,HW
        colored_B = torch.bmm(X_B, whitend).unsqueeze(3).view(B, C, H, -1)  # B,C,H,W

        return self.alpha * (colored_B + s_B_mu) + (1 - self.alpha) * c_A, cov_c, eigen_s


class Decoder(nn.Module):
    def __init__(self, input_dim, mask, n_group, bias_dim, mlp_dim, repeat_num=4,
                 norm='ln', device=None):
        super(Decoder, self).__init__()

        curr_dim = input_dim

        # Bottleneck layers
        self.resblocks = nn.ModuleList(
            [ResidualBlock(dim=curr_dim, norm='none', n_group=n_group) for i in range(repeat_num)])
        self.gdwct_modules = nn.ModuleList(
            [WCT(n_group, device, input_dim, mlp_dim, bias_dim, mask) for i in range(repeat_num + 1)])

        # Up-sampling layers
        layers = []
        for i in range(2):
            layers += [Upsample(scale_factor=2, mode='nearest')]
            layers += [ConvBlock(curr_dim, curr_dim // 2, 5, 1, 2, norm=norm, n_group=n_group)]
            curr_dim = curr_dim // 2

        layers += [ConvBlock(curr_dim, 3, 7, 1, 3, norm='none', activation='tanh')]

        self.main = nn.Sequential(*layers)

    def forward(self, c_A, s_B):
        whitening_reg = []
        coloring_reg = []

        # Multi-hops
        for i, resblock in enumerate(self.resblocks):
            if i == 0:
                c_A, cov, eigen_s = self.gdwct_modules[i](c_A, s_B)
                whitening_reg += [cov]
                coloring_reg += [eigen_s]

            c_A = resblock(c_A)
            c_A, cov, eigen_s = self.gdwct_modules[i + 1](c_A, s_B)
            whitening_reg += [cov]
            coloring_reg += [eigen_s]

        # cov_reg: G,C//G,C//G
        # W_reg: B*G,C//G,C//G
        return self.main(c_A), whitening_reg, coloring_reg