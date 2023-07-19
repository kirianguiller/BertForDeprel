from torch import Tensor
from .BiAffineTrankit import FixedClassDeepBiAffineClassifier
from .BertForDepRelOutput import BertForDeprelBatchOutput


from torch.nn import Linear, Module


class PosAndDeprelParserHead(Module):
    def __init__(self, n_uposs: int, n_deprels: int, n_feats: int, n_lemma_scripts: int, n_xposs: int, llm_output_size: int):
        super(PosAndDeprelParserHead, self).__init__()

        # Arc and label
        self.down_dim = llm_output_size // 4
        self.down_projection = Linear(llm_output_size, self.down_dim)
        self.arc = FixedClassDeepBiAffineClassifier(self.down_dim, self.down_dim,
                                       self.down_dim, 1)
        self.deprel = FixedClassDeepBiAffineClassifier(self.down_dim, self.down_dim,
                                    self.down_dim, n_deprels)

        # Label POS
        self.uposs_ffn = Linear(llm_output_size, n_uposs)
        self.xposs_ffn = Linear(llm_output_size, n_xposs)
        self.feats_ffn = Linear(llm_output_size, n_feats)
        self.lemma_scripts_ffn = Linear(llm_output_size, n_lemma_scripts)


    def forward(self, x: Tensor) -> BertForDeprelBatchOutput:
        uposs = self.uposs_ffn(x)
        xposs = self.xposs_ffn(x)
        feats = self.feats_ffn(x)
        lemma_scripts = self.lemma_scripts_ffn(x)
        down_projection_embedding = self.down_projection(x) # torch.Size([16, 28, 256])
        # Predicting binary relations among all words, so x and y are the same arg
        arc_scores = self.arc(down_projection_embedding, down_projection_embedding) # torch.Size([16, 28, 28, 1])
        deprel_scores = self.deprel(down_projection_embedding, down_projection_embedding) # torch.Size([16, 28, 28, 40])
        heads = arc_scores.squeeze(3)
        deprels = deprel_scores.permute(0, 3, 2, 1)

        return BertForDeprelBatchOutput(
            uposs=uposs,
            xposs=xposs,
            feats=feats,
            lemma_scripts=lemma_scripts,
            deprels=deprels,
            heads=heads,
        )
