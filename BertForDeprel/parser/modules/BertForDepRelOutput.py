import torch


from typing import TypedDict


class BertForDeprelOutput(TypedDict):
    uposs: torch.Tensor
    xposs: torch.Tensor
    feats: torch.Tensor
    lemma_scripts: torch.Tensor
    deprels: torch.Tensor
    heads: torch.Tensor
