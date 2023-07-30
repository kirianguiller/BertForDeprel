from dataclasses import dataclass
import torch


@dataclass
class BertForDeprelSentenceOutput:
    """Each prediction tensor has size (T, C), where T is the maximum sequence length for the containing batch,
    and C is the number of classes being assigned a probability for the particular tensor."""
    uposs: torch.Tensor
    xposs: torch.Tensor
    feats: torch.Tensor
    lemma_scripts: torch.Tensor
    deprels: torch.Tensor
    heads: torch.Tensor

    # 1 if sequence token begins a new word, 0 otherwise. Size is (B, T).
    subwords_start: torch.Tensor
    # Maps word index + 1 to the index in the sequence_token_ids where the word begins. Size is (W).
    idx_converter: torch.Tensor


@dataclass
class BertForDeprelBatchOutput:
    """Each prediction tensor has size (B, T, C), where B is the number of sentences in the batch."""
    uposs: torch.Tensor
    xposs: torch.Tensor
    feats: torch.Tensor
    lemma_scripts: torch.Tensor
    deprels: torch.Tensor
    heads: torch.Tensor

    # 1 if sequence token begins a new word, 0 otherwise. Size is (B, T).
    subwords_start: torch.Tensor
    # Maps word index + 1 to the index in the sequence_token_ids where the word begins. Size is (B, W).
    idx_converter: torch.Tensor


    def detach(self):
        """Return a new result with all of the Tensors detached from backprop (used for prediction)."""
        return BertForDeprelBatchOutput(
            uposs=self.uposs.detach(),
            xposs=self.xposs.detach(),
            feats=self.feats.detach(),
            lemma_scripts=self.lemma_scripts.detach(),
            deprels=self.deprels.detach(),
            heads=self.heads.detach(),
            subwords_start=self.subwords_start,
            idx_converter=self.idx_converter
        )

    def distributions_for_sentence(self, sentence_idx: int) -> BertForDeprelSentenceOutput:
        """Return the model output for the sentence at the specified index."""
        return BertForDeprelSentenceOutput(
            heads=self.heads[sentence_idx].clone(),
            deprels=self.deprels[sentence_idx].clone(),
            uposs=self.uposs[sentence_idx].clone(),
            xposs=self.xposs[sentence_idx].clone(),
            feats= self.feats[sentence_idx].clone(),
            lemma_scripts=self.lemma_scripts[sentence_idx].clone(),
            subwords_start = self.subwords_start[sentence_idx],
            idx_converter = self.idx_converter[sentence_idx]
        )
