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


    def detach(self):
        """Return a new result with all of the Tensors detached from backprop (used for prediction)."""
        return BertForDeprelOutput(
            uposs=self.uposs.detach(),
            xposs=self.xposs.detach(),
            feats=self.feats.detach(),
            lemma_scripts=self.lemma_scripts.detach(),
            deprels=self.deprels.detach(),
            heads=self.heads.detach()
        )
