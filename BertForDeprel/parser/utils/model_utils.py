from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from torch import nn
from parser.utils.modules_utils import MLP, BiAffine


class BertForDeprel(nn.Module):

    def __init__(self, args):
        super(BertForDeprel, self).__init__()
        self.args = args
        self.bert_layer = AutoModel.from_pretrained(self.args.bert_type)

        #Freeze bert layers
        if self.args.freeze_bert:
            self.freeze_bert()


        if self.args.reinit_bert:
            self.bert_layer.init_weights()

        # MLP and Bi-Affine layers
        mlp_input = args.mlp_input
        mlp_input = self.bert_layer.config.hidden_size #expected to get embedding size of bert custom model
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

        # TODO_LEMMA : here we need to add a classifier (MLP) that has an input shape of ...
        # ... 'mlp_input' and an output size of 'len(set(all_lemma_scripts))'.
        # !!! Important : we need to get the set of all lemma scripts in our dataset so we ... 
        # ... can create a classifier with the good number of class


        # Arc MLPs
        self.arc_mlp_h = MLP(mlp_input, mlp_arc_hidden, mlp_layers, 'ReLU', mlp_dropout)
        self.arc_mlp_d = MLP(mlp_input, mlp_arc_hidden, mlp_layers, 'ReLU', mlp_dropout)
        # Label MLPs
        self.lab_mlp_h = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
        self.lab_mlp_d = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
        # Label POS
        self.pos_mlp = MLP(mlp_input, n_pos, mlp_pos_layers, 'ReLU', mlp_dropout)

        # Label lemma_script
        self.lemma_script_mlp = MLP(mlp_input, n_lemma_scripts, mpl_lemma_scripts_layers, 'ReLU', mlp_dropout)
        
        # self.pos_mlp = MLP(mlp_input, n_lemma_rules, mlp_pos_layers, 'ReLU', mlp_dropout)

        # BiAffine layers
        self.arc_biaffine = BiAffine(mlp_arc_hidden, 1)
        self.lab_biaffine = BiAffine(mlp_lab_hidden, n_labels_main)


        
        # Label secondary MLPs
        if args.split_deprel:
            n_labels_aux = len(args.list_deprel_aux)  
            self.lab_aux_mlp_h = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
            self.lab_aux_mlp_d = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
            self.lab_aux_biaffine = BiAffine(mlp_lab_hidden, n_labels_aux)


    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        
        bert_output : BaseModelOutputWithPoolingAndCrossAttentions = self.bert_layer(seq, attention_mask = attn_masks)

        x = bert_output.last_hidden_state 

        # deprecated : if transformers library is 4.4.0 or above, the output is not a tuple but a 
        # ... complex object `BaseModelOutputWithPoolingAndCrossAttentions`

        if type(x)==tuple:
            x = x[0]

        arc_h = self.arc_mlp_h(x)
        arc_d = self.arc_mlp_d(x)
        lab_h = self.lab_mlp_h(x)
        lab_d = self.lab_mlp_d(x)

        # TODO_LEMMA : here, add the lemma prediction and return it :)

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


    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.5)
        else:# isinstance(module, LayerNorm):
            try:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            except:
                pass
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def freeze_bert(self):
        for p in self.bert_layer.parameters():
            p.requires_grad = False

        print("Bert layers freezed")
