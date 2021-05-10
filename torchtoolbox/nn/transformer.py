__all__ = ['PatchEmbedding', 'PositionEncoding', 'FeedForward']

import math

import torch
from torch import nn

from ..tools import check_twin
from .activation import Activation


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, dim, in_channels=3, norm_layer=nn.LayerNorm, out_order=('B', 'SL', 'D')):
        super().__init__()
        assert out_order in (('B', 'SL', 'D'), ('SL', 'B', 'D'))
        self.batch_first = True if out_order[0] == 'B' else False

        img_size = check_twin(img_size)
        patch_size = check_twin(patch_size)

        patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = patch_grid[0] * patch_grid[1]

        self.dump_patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.dump_patch(x).flatten(2)
        if self.batch_first:
            x = x.transpose(1, 2)
        else:
            x = x.permute(2, 0, 1)
        x = self.norm(x)
        return x


class PositionEncoding(nn.Module):
    def __init__(self, sequence_length, dim, dropout=0., learnable=False, batch_axis=0):
        super().__init__()
        self.learnable = learnable

        if not learnable:
            pe = torch.zeros(sequence_length, dim)
            position = torch.arange(0, sequence_length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Parameter(torch.Tensor(sequence_length, dim))
            nn.init.normal_(self.pe, std=0.02)

        self.pe.unsqueeze_(batch_axis)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x + self.pe)
        return x

    def no_wd(self, decay: list, no_decay: list):
        if self.learnable:
            no_decay.append(self.pe)

    def num_param(self):
        if self.learnable:
            return self.pe.numel(), 0
        else:
            return 0, 0


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, activation='gelu', dropout=0.):
        super().__init__()
        # do not add last dropout, if need add after this.
        self.ffn = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 Activation(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim))  # yapf:disable

    def forward(self, x):
        return self.ffn(x)
