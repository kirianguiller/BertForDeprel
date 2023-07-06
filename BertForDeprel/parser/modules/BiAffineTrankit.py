# Implementation from https://github.com/nlp-uoregon/trankit/blob/master/trankit/models/base_models.py
import torch
from torch import nn


def _mlp(in_dimension, out_dimension):
    """Using an MLP on the LLM outputs creates what the authors call a "deep bilinear attention mechanism";
    See equations 4, 5 and 6 for an example (MLP equations for fixed class classifier not shown in the paper
    explicitly, but the idea is the same)."""
    return nn.Sequential(
        nn.Linear(in_dimension, out_dimension),
        nn.ReLU(),
        nn.Dropout(0.5)
    )

class FixedClassDeepBiAffineClassifier(nn.Module):
    '''
    Based on equation 3 from Dozat and Manning, 2016: Deep Biafine Attention for Neural Dependency Parsing
    (https://arxiv.org/abs/1611.01734).
    Note that the fixed class classifier is more general than the variable class classifier, and can be used
    for both arc scoring and arc labeling. This is arguably more powerful than what's presented in the paper;
    equation 2 (the variable class biaffine classifier) models the probability of words i and j entering a
    head-dependent relationship as well as the prior probability of word i having dependents, but it doesn't
    model the prior probability of word j having a governing head, which could be important for finding the
    root node. The bias term in equation 3 is redundant in the arc scoring context, since we always have arcs
    and don't need a prior probability on their existence.
    '''

    def __init__(self, in_dim1, in_dim2, hidden_dim, output_dim):
        super().__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.ffn1 = _mlp(in_dim1, hidden_dim)
        self.ffn2 = _mlp(in_dim2, hidden_dim)
        # Weights for the biaffine scorer; +1 for the biases
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
