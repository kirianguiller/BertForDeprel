import os
from timeit import default_timer as timer
from typing import Optional
import numpy as np
import json

import torch
from torch.optim import AdamW
from torch.nn import Module, CrossEntropyLoss, Linear

from transformers import AutoModel, XLMRobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import AdapterConfig

from .load_data_utils import SequenceBatch_T # this one is specific to adapter-transformers (https://github.com/adapter-hub/adapter-transformers)
from .modules_utils import BiAffineTrankit
from .scores_and_losses_utils import compute_loss_head, compute_loss_deprel, compute_loss_poss, confusion_matrix, compute_acc_deprel, compute_acc_head, compute_acc_pos, compute_LAS, compute_LAS_chuliu, compute_LAS
from .chuliu_edmonds_utils import chuliu_edmonds_one_root
from .types import ModelParams_T

class PosAndDeprelParserHead(Module):
    def __init__(self, n_upos: int, n_deprels: int, llm_output_size: int):
        super(PosAndDeprelParserHead, self).__init__()

        # Arc and label
        self.down_dim = llm_output_size // 4
        self.down_projection = Linear(llm_output_size, self.down_dim)
        self.arc = BiAffineTrankit(self.down_dim, self.down_dim,
                                       self.down_dim, 1)
        self.deprel = BiAffineTrankit(self.down_dim, self.down_dim,
                                    self.down_dim, n_deprels)
        
        # Label POS
        self.upos_ffn = Linear(llm_output_size, n_upos)
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
    def __init__(self, model_params: ModelParams_T, pretrain_model_params: Optional[ModelParams_T] = None):
        super(BertForDeprel, self).__init__()
        self.model_params = model_params
        self.pretrain_model_params = pretrain_model_params
        self.llm_layer: XLMRobertaModel = AutoModel.from_pretrained(model_params["embedding_type"], cache_dir=model_params.get("embedding_cached_path", None))
        llm_hidden_size = self.llm_layer.config.hidden_size #expected to get embedding size of bert custom model
        adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=4)
        # TODO : find better name (tagger)
        adapter_name = model_params["model_name"]
        self.llm_layer.add_adapter(adapter_name, config=adapter_config)
        self.llm_layer.train_adapter([adapter_name])
        self.llm_layer.set_active_adapters([adapter_name]) 

        n_uposs = len(model_params["annotation_schema"]["uposs"])
        n_deprels = len(model_params["annotation_schema"]["deprels"])
        self.tagger_layer = PosAndDeprelParserHead(n_uposs, n_deprels, llm_hidden_size)

        if self.pretrain_model_params:
            print("Loading weights of the pretrained model")
            self.load_pretrained()

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


    def train_epoch(self, loader, device):
        start = timer()
        self.train()
        batch: SequenceBatch_T
        for n_batch, batch in enumerate(loader):
            seq_ids = batch["seq_ids"]
            attn_masks = batch["attn_masks"]
            heads = batch["heads"]
            deprels = batch["deprels"]
            poss = batch["uposs"]
            
            self.optimizer.zero_grad()
            seq_ids, attn_masks, heads_true, deprels_true, poss_true = seq_ids.to(device), attn_masks.to(device), heads.to(device), deprels.to(device), poss.to(device)

            heads_pred, deprels_pred, poss_pred = self.forward(seq_ids, attn_masks)
            
            loss_batch = compute_loss_head(heads_pred, heads_true, self.criterion)
            loss_batch += compute_loss_deprel(deprels_pred, deprels_true, heads_true.clone(), self.criterion)
            loss_batch += compute_loss_poss(poss_pred, poss_true, self.criterion)
            
            loss_batch.backward()
            self.optimizer.step()

            print(
            f'Training: {100 * (n_batch + 1) / len(loader):.2f}% complete. {timer() - start:.2f} seconds in epoch',
            end='\r')

    def eval_epoch(self, loader, device):
        self.eval()
        with torch.no_grad():
            loss_head_epoch = 0.0
            loss_deprel_epoch = 0.0
            loss_poss_epoch = 0
            good_head_epoch, total_head_epoch = 0.0, 0.0
            good_pos_epoch, total_pos_epoch = 0.0, 0.0
            good_deprel_epoch, total_deprel_epoch = 0.0, 0.0
            n_correct_LAS_epoch, n_correct_LAS_epoch,n_correct_LAS_aux_epoch, n_total_epoch = 0.0, 0.0, 0.0, 0.0
            n_correct_LAS_chuliu_epoch, n_correct_LAS_chuliu_main_epoch,n_correct_LAS_chuliu_aux_epoch, n_total_epoch = 0.0, 0.0, 0.0, 0.0
            
            batch: SequenceBatch_T
            for n_batch, batch in enumerate(loader):
                seq_ids = batch["seq_ids"]
                attn_masks = batch["attn_masks"]
                subwords_start = batch["subwords_start"]
                idx_convertor = batch["idx_convertor"]
                heads = batch["heads"]
                deprels = batch["deprels"]
                poss = batch["uposs"]

                print(f"evaluation on the dataset ... {n_batch}/{len(loader)}batches", end="\r")
                seq_ids, attn_masks, heads_true, deprels_true, poss_true = seq_ids.to(device), attn_masks.to(device), heads.to(device), deprels.to(device), poss.to(device)
                heads_pred, deprels_pred, poss_pred = self.forward(seq_ids, attn_masks)
                
                heads_pred, deprels_pred, poss_pred = heads_pred.detach(), deprels_pred.detach(), poss_pred.detach()

                chuliu_heads_pred = heads_true.clone()
                for i_vector, (heads_pred_vector, subwords_start_vector, idx_convertor_vector) in enumerate(zip(heads_pred, subwords_start, idx_convertor)):
                    subwords_start_with_root = subwords_start_vector.clone()
                    subwords_start_with_root[0] = True

                    heads_pred_np = heads_pred_vector[:,subwords_start_with_root == 1][subwords_start_with_root == 1]
                    heads_pred_np = heads_pred_np.cpu().numpy()
                    
                    chuliu_heads_vector = chuliu_edmonds_one_root(np.transpose(heads_pred_np, (1,0)))[1:]
                    
                    for i_token, chuliu_head_pred in enumerate(chuliu_heads_vector):
                        chuliu_heads_pred[i_vector, idx_convertor_vector[i_token+1]] = idx_convertor_vector[chuliu_head_pred]
                    
                n_correct_LAS_batch, n_correct_LAS_batch, n_total_batch = \
                    compute_LAS(heads_pred, deprels_pred, heads_true, deprels_true)
                n_correct_LAS_chuliu_batch, n_correct_LAS_chuliu_main_batch, n_total_batch = \
                    compute_LAS_chuliu(chuliu_heads_pred, deprels_pred, heads_true, deprels_true)
                n_correct_LAS_epoch += n_correct_LAS_batch
                n_correct_LAS_chuliu_epoch += n_correct_LAS_chuliu_batch
                n_total_epoch += n_total_batch

                loss_head_batch = compute_loss_head(heads_pred, heads_true, self.criterion)
                good_head_batch, total_head_batch = compute_acc_head(heads_pred, heads_true, eps=0)
                loss_head_epoch += loss_head_batch.item()
                good_head_epoch += good_head_batch
                total_head_epoch += total_head_batch
                
                loss_deprel_batch = compute_loss_deprel(deprels_pred, deprels_true, heads_true, self.criterion)
                good_deprel_batch, total_deprel_batch = compute_acc_deprel(deprels_pred, deprels_true, heads_true, eps=0)
                loss_deprel_epoch += loss_deprel_batch.item()
                good_deprel_epoch += good_deprel_batch
                total_deprel_epoch += total_deprel_batch
                n_correct_LAS_epoch += n_correct_LAS_batch
                
                good_pos_batch, total_pos_batch = compute_acc_pos(poss_pred, poss_true, eps=0)
                good_pos_epoch += good_pos_batch
                total_pos_epoch += total_pos_batch

                loss_poss_batch = compute_loss_poss(poss_pred, poss_true, self.criterion)
                loss_poss_epoch += loss_poss_batch


            loss_head_epoch = loss_head_epoch/len(loader)
            acc_head_epoch = good_head_epoch/total_head_epoch
            
            loss_deprel_epoch = loss_deprel_epoch/len(loader)
            acc_deprel_epoch = good_deprel_epoch/total_deprel_epoch
            
            acc_pos_epoch = good_pos_epoch/total_pos_epoch

            LAS_epoch = n_correct_LAS_epoch/n_total_epoch
            LAS_chuliu_epoch = n_correct_LAS_chuliu_epoch/n_total_epoch
            LAS_epoch = n_correct_LAS_epoch/n_total_epoch


            loss_epoch = loss_head_epoch + loss_deprel_epoch + loss_poss_epoch 
            print("\nevaluation result: LAS={:.3f}; LAS_chuliu={:.3f}; loss_epoch={:.3f}; eval_acc_head={:.3f}; eval_acc_deprel = {:.3f}, eval_acc_pos = {:.3f}\n".format(
            LAS_epoch, LAS_chuliu_epoch, loss_epoch, LAS_epoch, acc_head_epoch, acc_pos_epoch))

        results = {
            "LAS_epoch": LAS_epoch,
            "LAS_chuliu_epoch": LAS_chuliu_epoch,
            "LAS_epoch": LAS_epoch,
            "acc_head_epoch": acc_head_epoch,
            "acc_deprel_epoch" : acc_deprel_epoch,
            "acc_pos_epoch": acc_pos_epoch,
            "loss_head_epoch": loss_head_epoch,
            "loss_deprel_epoch": loss_deprel_epoch,
            "loss_epoch": loss_epoch,
        }

        return results
        
        
    def save_model(self, epoch):
        trainable_weight_names = [n for n, p in self.llm_layer.named_parameters() if p.requires_grad] + \
                                 [n for n, p in self.tagger_layer.named_parameters() if p.requires_grad]
        state = {"adapter": {}, "tagger": {}, "epoch": epoch}
        for k, v in self.llm_layer.state_dict().items():
            if k in trainable_weight_names:
                state["adapter"][k] = v
        for k, v in self.tagger_layer.state_dict().items():
            if k in trainable_weight_names:
                state["tagger"][k] = v

        ckpt_fpath = os.path.join(self.model_params["root_folder_path"], self.model_params["model_name"] + ".pt")
        config_path = os.path.join(self.model_params["root_folder_path"], self.model_params["model_name"] + ".config.json")
        torch.save(state, ckpt_fpath)
        print(
            "Saving adapter weights to ... {} ({:.2f} MB) (conf path : {})".format(
                ckpt_fpath, os.path.getsize(ckpt_fpath) * 1.0 / (1024 * 1024),
                config_path
            )
        )
        with open(config_path, "w") as outfile:
            outfile.write(json.dumps(self.model_params))

    def load_pretrained(self):
        params = self.pretrain_model_params or self.model_params
        ckpt_fpath = os.path.join(params["root_folder_path"], params["model_name"] + ".pt")
        checkpoint_state = torch.load(ckpt_fpath)

        self.tagger_layer.load_state_dict(checkpoint_state["tagger"])
        
        model_dict = self.llm_layer.state_dict()
        for layer_name, weights in checkpoint_state["adapter"].items():
            if self.pretrain_model_params:
                layer_name = layer_name.replace(self.pretrain_model_params["model_name"], self.model_params["model_name"])
            if layer_name in model_dict:
                model_dict[layer_name] = weights
        self.llm_layer.load_state_dict(model_dict)
        return

### To reactivate if probleme in the loading of the model states
# loaded_state_dict = OrderedDict()
# for k, v in checkpoint["state_dict"].items():
#     name = k.replace("module.", "")
#     loaded_state_dict[name] = v