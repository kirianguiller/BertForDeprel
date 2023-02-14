from timeit import default_timer as timer

from transformers import AutoModel, XLMRobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import AdapterConfig # this one is specific to adapter-transformers (https://github.com/adapter-hub/adapter-transformers)

from torch import Tensor
from torch.optim import AdamW
from torch.nn import Module, CrossEntropyLoss, Linear

from .modules_utils import BiAffineTrankit
from .scores_and_losses_utils import compute_loss_head, compute_loss_deprel, compute_loss_poss

class PosAndDeprelParserHead(Module):
    def __init__(self, args, input_size: int):
        super(PosAndDeprelParserHead, self).__init__()
        args = args
        self.args = args
        mlp_input = input_size 
        n_labels_main = len(args.list_deprel_main)
        n_pos = len(args.list_pos)

        # Arc and label
        self.down_dim = mlp_input // 4
        self.down_projection = Linear(mlp_input, self.down_dim)
        self.arc = BiAffineTrankit(self.down_dim, self.down_dim,
                                       self.down_dim, 1)
        self.deprel = BiAffineTrankit(self.down_dim, self.down_dim,
                                    self.down_dim, n_labels_main)
        
        # Label POS
        self.upos_ffn = Linear(mlp_input, n_pos)
        # self.xpos_ffn = Linear(self.xlmr_dim + 50, len(self.vocabs[XPOS]))
        # self.feats_ffn = Linear(self.xlmr_dim, len(self.vocabs[FEATS]))
        

    def forward(self, x):
        pos = self.upos_ffn(x)
        down_projection_embedding = self.down_projection(x) # torch.Size([16, 28, 256])
        arc_scores = self.arc(down_projection_embedding, down_projection_embedding) # torch.Size([16, 28, 28, 1])
        deprel_scores = self.deprel(down_projection_embedding, down_projection_embedding) # torch.Size([16, 28, 28, 40])
        arc_scores = arc_scores.squeeze(3)
        deprel_scores = deprel_scores.permute(0, 3, 2, 1)
        return arc_scores, deprel_scores, pos


class BertForDeprel(Module):
    def __init__(self, args):
        super(BertForDeprel, self).__init__()
        self.args = args
        self.llm_layer: XLMRobertaModel = AutoModel.from_pretrained(self.args.bert_type)
        llm_hidden_size = self.llm_layer.config.hidden_size #expected to get embedding size of bert custom model
        adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=4)
        # TODO : find better name (tagger)
        adapter_name = "tagger"
        self.llm_layer.add_adapter(adapter_name, config=adapter_config)
        self.llm_layer.train_adapter([adapter_name])
        self.llm_layer.set_active_adapters([adapter_name]) 
        self.tagger_layer = PosAndDeprelParserHead(args, llm_hidden_size)
        self.total_trainable_parameters = self.get_total_trainable_parameters()
        print("TOTAL TRAINABLE PARAMETERS : ", self.total_trainable_parameters)

        self._set_criterions_and_optimizer()


    def _set_criterions_and_optimizer(self):
        self.criterion = CrossEntropyLoss(ignore_index=-1)
        self.optimizer = AdamW(self.parameters(), lr=0.00005)
        print("Criterion and Optimizers set")

    def get_total_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        #Feeding the input to BERT model to obtain contextualized representations
        bert_output : BaseModelOutputWithPoolingAndCrossAttentions = self.llm_layer.forward(seq, attention_mask = attn_masks)

        x = bert_output.last_hidden_state 
        return self.tagger_layer(x)


    def train_epoch(self, train_loader, args):
        device = args.device
        start = timer()
        self.train()
        for n_batch, (seq, _, attn_masks, _, poss, heads, deprels_main) in enumerate(train_loader):
            self.optimizer.zero_grad()
            seq, attn_masks, heads_true, deprels_main_true, poss_true = seq.to(device), attn_masks.to(device), heads.to(device), deprels_main.to(device), poss.to(device)

            heads_pred, deprels_main_pred, poss_pred = self.forward(seq, attn_masks)
            
            loss_batch = compute_loss_head(heads_pred, heads_true, self.criterion)
            loss_batch += compute_loss_deprel(deprels_main_pred, deprels_main_true, heads_true.clone(), self.criterion)
            loss_batch += compute_loss_poss(poss_pred, poss_true, self.criterion)
            
            loss_batch.backward()
            self.optimizer.step()

            print(
            f'Training: {100 * (n_batch + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds in epoch',
            end='\r')
        
        