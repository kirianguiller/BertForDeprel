from transformers import AutoModel, BertModel, BertModelWithHeads, RobertaModel, XLMRobertaModel, AutoModelWithHeads, XLMRobertaModelWithHeads
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
# Following line are specific to adapter-transformers (https://github.com/adapter-hub/adapter-transformers)
from transformers import AdapterConfig
from torch import nn
from parser.utils.modules_utils import MLP, BiAffine


class PosAndDeprelParserHead(nn.Module):
    def __init__(self, args, input_size: int):
        super(PosAndDeprelParserHead, self).__init__()
        self.args = args
        # MLP and Bi-Affine layers
        mlp_input = input_size 
        mlp_arc_hidden = args.mlp_arc_hidden
        mlp_lab_hidden = args.mlp_lab_hidden
        mlp_dropout = args.mlp_dropout
        mlp_layers = args.mlp_layers
        mlp_pos_layers = args.mlp_pos_layers
        mpl_lemma_scripts_layers = args.mlp_pos_layers
        n_labels_main = len(args.list_deprel_main)
        n_pos = len(args.list_pos)
        n_lemma_scripts = len(args.list_lemma_script)
        print("\nKK len(args.list_lemma_script) ", len(args.list_lemma_script))

        # Arc MLPs
        self.arc_mlp_h = MLP(mlp_input, mlp_arc_hidden, mlp_layers, 'ReLU', mlp_dropout)
        self.arc_mlp_d = MLP(mlp_input, mlp_arc_hidden, mlp_layers, 'ReLU', mlp_dropout)
        # Label MLPs
        self.lab_mlp_h = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
        self.lab_mlp_d = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
        # Label POS
        self.pos_mlp = MLP(mlp_input, n_pos, mlp_pos_layers, 'ReLU', mlp_dropout)
        # self.upos_ffn = nn.Linear(mlp_input, n_pos)
        # self.xpos_ffn = nn.Linear(self.xlmr_dim + 50, len(self.vocabs[XPOS]))

        # Label lemma_script
        self.lemma_script_mlp = MLP(mlp_input, n_lemma_scripts, mpl_lemma_scripts_layers, 'ReLU', mlp_dropout)
        
        # self.pos_mlp = MLP(mlp_input, n_lemma_rules, mlp_pos_layers, 'ReLU', mlp_dropout)

        # BiAffine layers
        self.arc_biaffine = BiAffine(mlp_arc_hidden, 1, True, False)
        self.lab_biaffine = BiAffine(mlp_lab_hidden, n_labels_main, True, True)


        
        # Label secondary MLPs
        if args.split_deprel:
            n_labels_aux = len(args.list_deprel_aux)  
            self.lab_aux_mlp_h = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
            self.lab_aux_mlp_d = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
            self.lab_aux_biaffine = BiAffine(mlp_lab_hidden, n_labels_aux)

    def forward(self, x):
        arc_h = self.arc_mlp_h(x)
        arc_d = self.arc_mlp_d(x)
        lab_h = self.lab_mlp_h(x)
        lab_d = self.lab_mlp_d(x)

        pos = self.pos_mlp(x)
        lemma_script = self.lemma_script_mlp(x)

        S_arc = self.arc_biaffine(arc_h, arc_d)
        S_lab = self.lab_biaffine(lab_h, lab_d)

        if self.args.split_deprel:
            lab_aux_h = self.lab_aux_mlp_h(x)
            lab_aux_d = self.lab_aux_mlp_d(x)
            S_lab_aux = self.lab_aux_biaffine(lab_h, lab_d)
        else:
            S_lab_aux = S_lab.clone()

        # return twice S_lab for replacing S_lab_aux and always having 4 elements in the output
        return S_arc, S_lab, S_lab_aux, pos, lemma_script

class BertForDeprel(nn.Module):

    def __init__(self, args):
        super(BertForDeprel, self).__init__()
        self.args = args
        # self.llm_layer: XLMRobertaModel = AutoModel.from_pretrained(self.args.bert_type)
        self.llm_layer: BertModelWithHeads = AutoModelWithHeads.from_pretrained(self.args.bert_type)
        
        adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=4)
        # TODO : find better name (tagger)
        adapter_name = "tagger"
        self.llm_layer.add_adapter(adapter_name, config=adapter_config)
        self.llm_layer.train_adapter([adapter_name])
        self.llm_layer.set_active_adapters([adapter_name]) 

        bert_hidden_size = self.llm_layer.config.hidden_size #expected to get embedding size of bert custom model
        self.tagger_layer = PosAndDeprelParserHead(args, bert_hidden_size)
        #Freeze bert layers
        # if self.args.freeze_bert:
        #     self.freeze_bert()

        # if self.args.reinit_bert:
        #     self.llm_layer.init_weights()

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


    # def init_weights(self, module):
    #     """ ReInitialize the weights """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=0.5)
    #     else:# isinstance(module, LayerNorm):
    #         try:
    #             module.bias.data.zero_()
    #             module.weight.data.fill_(1.0)
    #         except:
    #             pass
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()


    # def freeze_bert(self):
    #     for p in self.llm_layer.parameters():
    #         p.requires_grad = False

    #     print("Bert layers freezed")
