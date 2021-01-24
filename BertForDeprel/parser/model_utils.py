from transformers import BertModel, AutoModel
from transformers import CamembertModel
from torch import nn
from parser.utils.modules_utils import MLP, BiAffine, BiAffine2, BiLSTM
# from modules_utils import MLP, BiAffine, BiAffine2, BiLSTM


class BertForDeprel(nn.Module):

    # def __init__(self, n_labels_main=1, n_labels_aux=False,freeze_bert = False, reinit_bert=False, criterion = False, bert_language="english"):
    def __init__(self, args):
        super(BertForDeprel, self).__init__()
        self.args = args
        #Instantiating BERT model object
        if self.args.bert_type == "bert":
            self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        elif self.args.bert_type == "camembert":
            self.bert_layer = CamembertModel.from_pretrained('camembert-base')
        elif self.args.bert_type == "mbert":
            self.bert_layer = BertModel.from_pretrained('bert-base-multilingual-uncased')
        else:
            self.bert_layer = AutoModel.from_pretrained(self.args.bert_type)
            # assert Exception("You must choose `bert_language` as `french` or `english`")

        #Freeze bert layers
        if self.args.freeze_bert:
            self.freeze_bert()


        if self.args.reinit_bert:
            self.bert_layer.init_weights()

        # n_embed_lstm = 200
        # n_lstm_hidden = 100
        # n_lstm_layers = 3
        # lstm_dropout = 0.3
        # # the word-lstm layer
        # self.lstm = BiLSTM(input_size=n_embed_lstm,
        #                    hidden_size=n_lstm_hidden,
        #                    num_layers=n_lstm_layers,
        #                    dropout=lstm_dropout)

        # MLP and Bi-Affine layers


        mlp_input = args.mlp_input
        mlp_input = self.bert_layer.config.hidden_size #expected to get embedding size of bert custom model
        mlp_arc_hidden = args.mlp_arc_hidden
        mlp_lab_hidden = args.mlp_lab_hidden
        mlp_dropout = args.mlp_dropout
        mlp_layers = args.mlp_layers
        mlp_pos_layers = args.mlp_pos_layers
        n_labels_main = len(args.list_deprel_main)
        n_pos = len(args.list_pos) + 1
        # Arc MLPs
        self.arc_mlp_h = MLP(mlp_input, mlp_arc_hidden, mlp_layers, 'ReLU', mlp_dropout)
        self.arc_mlp_d = MLP(mlp_input, mlp_arc_hidden, mlp_layers, 'ReLU', mlp_dropout)
        # Label MLPs
        self.lab_mlp_h = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
        self.lab_mlp_d = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
        # Label POS
        self.pos_mlp = MLP(mlp_input, n_pos, mlp_pos_layers, 'ReLU', mlp_dropout)

        # BiAffine layers
        self.arc_biaffine = BiAffine(mlp_arc_hidden, 1)
        self.lab_biaffine = BiAffine(mlp_lab_hidden, n_labels_main)
        
        # Label secondary MLPs
        if args.split_deprel:
            n_labels_aux = len(args.list_deprel_aux)  
            self.lab_aux_mlp_h = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
            self.lab_aux_mlp_d = MLP(mlp_input, mlp_lab_hidden, mlp_layers, 'ReLU', mlp_dropout)
            self.lab_aux_biaffine = BiAffine(mlp_lab_hidden, n_labels_aux)



        # # the Biaffine layers
        # self.arc_attn = BiAffine2(n_in=mlp_arc_hidden,
        #                          bias_x=True,
        #                          bias_y=False)
        # self.rel_attn = BiAffine2(n_in=mlp_lab_hidden,
        #                          n_out=num_labels,
        #                          bias_x=True,
        #                          bias_y=True)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        
        x = self.bert_layer(seq, attention_mask = attn_masks)
        if type(x)==tuple:
            x = x[0]
        # print("x before lstm", x.size())

        # x, _ = self.lstm(x)
        # print("x after lstm", x.size())

        arc_h = self.arc_mlp_h(x)
        arc_d = self.arc_mlp_d(x)
        lab_h = self.lab_mlp_h(x)
        lab_d = self.lab_mlp_d(x)


        pos = self.pos_mlp(x)

        S_arc = self.arc_biaffine(arc_h, arc_d)
        S_lab = self.lab_biaffine(lab_h, lab_d)

        if self.args.split_deprel:
            lab_aux_h = self.lab_aux_mlp_h(x)
            lab_aux_d = self.lab_aux_mlp_d(x)
            S_lab_aux = self.lab_aux_biaffine(lab_h, lab_d)
            return S_arc, S_lab, S_lab_aux, pos

        # return twice S_lab for replacing S_lab_aux and always having 4 elements in the output
        return S_arc, S_lab, S_lab.clone(), pos


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
