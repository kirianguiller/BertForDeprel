import torch
from torch import nn


class BiAffineTrankit(nn.Module):
    '''
    implemented based on the paper https://arxiv.org/abs/1611.01734
    '''

    def __init__(self, in_dim1, in_dim2, hidden_dim, output_dim):
        super().__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.ffn1 = nn.Sequential(
            nn.Linear(in_dim1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(in_dim2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # pairwise interactions
        self.pairwise_weight = nn.Parameter(torch.Tensor(in_dim1 + 1, in_dim2 + 1, output_dim))
        self.pairwise_weight.data.zero_()

    def forward(self, x1, x2):
        h1 = self.ffn1(x1)
        h2 = self.ffn2(x2)
        # make interactions
        g1 = torch.cat([h1, h1.new_ones(*h1.size()[:-1], 1)], len(h1.size()) - 1)
        g2 = torch.cat([h2, h2.new_ones(*h2.size()[:-1], 1)], len(h2.size()) - 1)

        g1_size = g1.size()
        g2_size = g2.size()

        g1_w = torch.mm(g1.view(-1, g1_size[-1]), self.pairwise_weight.view(-1, (self.in_dim2 + 1) * self.output_dim))
        g2 = g2.transpose(1, 2)
        g1_w_g2 = g1_w.view(g1_size[0], g1_size[1] * self.output_dim, g2_size[2]).bmm(g2)
        g1_w_g2 = g1_w_g2.view(g1_size[0], g1_size[1], self.output_dim, g2_size[1]).transpose(2, 3)
        return g1_w_g2
