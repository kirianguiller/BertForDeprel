# Shallow bilinear attention as described in https://arxiv.org/abs/1611.01734
import torch
from torch import nn


from math import prod


class BiAffine(nn.Module):
    """Biaffine attention layer."""
    def __init__(self, input_dim, output_dim, bias_x=False, bias_y=False):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.U = nn.Parameter(torch.FloatTensor(output_dim, input_dim + self.bias_x, input_dim + self.bias_y))
        nn.init.xavier_uniform_(self.U)


    def forward(self, Rh, Rd):
        if self.bias_x:
            Rh = torch.cat((Rh, torch.ones_like(Rh[..., :1])), -1)

        if self.bias_y:
            Rd = torch.cat((Rd, torch.ones_like(Rd[..., :1])), -1)

        Rh = Rh.unsqueeze(1)
        Rd = Rd.unsqueeze(1)

        S = Rh @ self.U @ Rd.transpose(-1, -2)

        return S.squeeze(1)


    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class BiAffine2(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)

        # print("x", x.size())
        # print("y", y.size())
        # s = x @ self.weight @ y.transpose(-1, -2)
        # print("s", s.size())
        # remove dim 1 if n_out == 1
        # s = s.unsqueeze(0)
        s = s.squeeze(1)

        return s
