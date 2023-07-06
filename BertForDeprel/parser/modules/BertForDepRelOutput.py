from dataclasses import dataclass
import torch


@dataclass
class BertForDeprelOutput:
    uposs: torch.Tensor
    xposs: torch.Tensor
    feats: torch.Tensor
    lemma_scripts: torch.Tensor
    deprels: torch.Tensor
    heads: torch.Tensor
