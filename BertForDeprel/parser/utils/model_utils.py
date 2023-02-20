import os
from timeit import default_timer as timer
from typing import Optional, TypedDict
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
from .scores_and_losses_utils import compute_loss_head, compute_loss_deprel, compute_loss_poss, confusion_matrix, compute_acc_deprel, compute_acc_head, compute_acc_upos, compute_LAS, compute_LAS_chuliu, compute_LAS
from .chuliu_edmonds_utils import chuliu_edmonds_one_root
from .types import ModelParams_T


class BertForDeprelOutput(TypedDict):
    uposs: torch.Tensor
    xposs: torch.Tensor
    feats: torch.Tensor
    lemma_scripts: torch.Tensor
    deprels: torch.Tensor
    heads: torch.Tensor

class PosAndDeprelParserHead(Module):
    def __init__(self, n_uposs: int, n_deprels: int, n_feats: int, n_lemma_scripts: int, n_xposs: int, llm_output_size: int):
        super(PosAndDeprelParserHead, self).__init__()

        # Arc and label
        self.down_dim = llm_output_size // 4
        self.down_projection = Linear(llm_output_size, self.down_dim)
        self.arc = BiAffineTrankit(self.down_dim, self.down_dim,
                                       self.down_dim, 1)
        self.deprel = BiAffineTrankit(self.down_dim, self.down_dim,
                                    self.down_dim, n_deprels)
        
        # Label POS
        self.uposs_ffn = Linear(llm_output_size, n_uposs)
        self.xposs_ffn = Linear(llm_output_size, n_xposs)
        self.feats_ffn = Linear(llm_output_size, n_feats)
        self.lemma_scripts_ffn = Linear(llm_output_size, n_lemma_scripts)
        

    def forward(self, x) -> BertForDeprelOutput:
        uposs = self.uposs_ffn(x)
        xposs = self.xposs_ffn(x)
        feats = self.feats_ffn(x)
        lemma_scripts = self.lemma_scripts_ffn(x)
        down_projection_embedding = self.down_projection(x) # torch.Size([16, 28, 256])
        arc_scores = self.arc(down_projection_embedding, down_projection_embedding) # torch.Size([16, 28, 28, 1])
        deprel_scores = self.deprel(down_projection_embedding, down_projection_embedding) # torch.Size([16, 28, 28, 40])
        heads = arc_scores.squeeze(3)
        deprels = deprel_scores.permute(0, 3, 2, 1)

        return {
            "uposs": uposs,
            "xposs": xposs,
            "feats": feats,
            "lemma_scripts": lemma_scripts,
            "deprels": deprels,
            "heads": heads,
        }





class BertForDeprel(Module):
    def __init__(self, model_params: ModelParams_T, pretrain_model_params: Optional[ModelParams_T] = None, overwrite_pretrain_classifiers = True):
        super(BertForDeprel, self).__init__()
        self.model_params = model_params
        self.pretrain_model_params = pretrain_model_params
        self.llm_layer: XLMRobertaModel = AutoModel.from_pretrained(model_params["embedding_type"])
        llm_hidden_size = self.llm_layer.config.hidden_size #expected to get embedding size of bert custom model
        adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=4)
        # TODO : find better name (tagger)
        adapter_name = model_params["model_name"]
        self.llm_layer.add_adapter(adapter_name, config=adapter_config)
        self.llm_layer.train_adapter([adapter_name])
        self.llm_layer.set_active_adapters([adapter_name]) 

        n_uposs = len(model_params["annotation_schema"]["uposs"])
        n_xposs = len(model_params["annotation_schema"]["xposs"])
        n_deprels = len(model_params["annotation_schema"]["deprels"])
        n_feats = len(model_params["annotation_schema"]["feats"])
        n_lemma_scripts = len(model_params["annotation_schema"]["lemma_scripts"])
        self.tagger_layer = PosAndDeprelParserHead(n_uposs, n_deprels, n_feats, n_lemma_scripts, n_xposs, llm_hidden_size)

        if self.pretrain_model_params:
            print("Loading weights of the pretrained model")
            self.load_pretrained(overwrite_pretrain_classifiers)

        self.total_trainable_parameters = self.get_total_trainable_parameters()
        print("TOTAL TRAINABLE PARAMETERS : ", self.total_trainable_parameters)

        self._set_criterions_and_optimizer()


    def _set_criterions_and_optimizer(self):
        self.criterion = CrossEntropyLoss(ignore_index=-1)
        self.optimizer = AdamW(self.parameters(), lr=0.00005)
        print("Criterion and Optimizers set")

    def get_total_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward(self, seq, attn_masks) -> BertForDeprelOutput:
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''
        #Feeding the input to BERT model to obtain contextualized representations
        bert_output : BaseModelOutputWithPoolingAndCrossAttentions = self.llm_layer.forward(seq, attention_mask = attn_masks)

        x = bert_output.last_hidden_state 
        output = self.tagger_layer.forward(x)
        return output


    def train_epoch(self, loader, device):
        time_from_start = 0
        parsing_speed = 0
        start = timer()
        self.train()
        processed_sentence_counter = 0
        total_number_batch = len(loader)
        print_every = max(1, total_number_batch // 8) # so we print only around 8 times per epochs
        batch: SequenceBatch_T
        for batch_counter, batch in enumerate(loader):
            seq_ids = batch["seq_ids"].to(device)
            attn_masks = batch["attn_masks"].to(device)
            heads_true = batch["heads"].to(device)
            deprels_true = batch["deprels"].to(device)
            uposs_true = batch["uposs"].to(device)
            xposs_true = batch["xposs"].to(device)
            feats_true = batch["feats"].to(device)
            lemma_scripts_true = batch["lemma_scripts"].to(device)
            
            self.optimizer.zero_grad()

            preds = self.forward(seq_ids, attn_masks)
            
            loss_batch = compute_loss_head(preds["heads"], heads_true, self.criterion)
            loss_batch += compute_loss_deprel(preds["deprels"], deprels_true, heads_true.clone(), self.criterion)
            loss_batch += compute_loss_poss(preds["uposs"], uposs_true, self.criterion)
            loss_batch += compute_loss_poss(preds["xposs"], xposs_true, self.criterion)
            loss_batch += compute_loss_poss(preds["feats"], feats_true, self.criterion)
            loss_batch += compute_loss_poss(preds["lemma_scripts"], lemma_scripts_true, self.criterion)
            
            loss_batch.backward()
            self.optimizer.step()

            processed_sentence_counter += seq_ids.size(0)
            time_from_start = timer() - start
            parsing_speed = int(round(((processed_sentence_counter + 1) / time_from_start) / 100, 2) * 100)
            
            if batch_counter % print_every == 0:
                print(
                f'Training: {100 * (batch_counter + 1) / len(loader):.2f}% complete. {time_from_start:.2f} seconds in epoch ({parsing_speed:.2f} sents/sec)',
                end='\r')
        print(f"\nFinished training epoch in {time_from_start:.2} seconds ({processed_sentence_counter} sentence at {parsing_speed} sents/sec)\n")

    def eval_epoch(self, loader, device):
        self.eval()
        with torch.no_grad():
            good_head_epoch, total_head_epoch, loss_head_epoch = 0.0, 0.0, 0.0
            good_uposs_epoch, total_uposs_epoch, loss_uposs_epoch = 0.0, 0.0, 0.0
            good_xposs_epoch, total_xposs_epoch, loss_xposs_epoch = 0.0, 0.0, 0.0
            good_feats_epoch, total_feats_epoch, loss_feats_epoch = 0.0, 0.0, 0.0
            good_lemma_scripts_epoch, total_lemma_scripts_epoch, loss_lemma_scripts_epoch = 0.0, 0.0, 0.0
            good_deprel_epoch, total_deprel_epoch, loss_deprel_epoch = 0.0, 0.0, 0.0
            n_correct_LAS_epoch, n_correct_LAS_epoch, n_total_epoch = 0.0, 0.0, 0.0
            n_correct_LAS_chuliu_epoch, n_total_epoch = 0.0, 0.0
            
            start = timer()
            processed_sentence_counter = 0
            total_number_batch = len(loader)
            print_every = max(1, total_number_batch // 4) # so we print only around 8 times per epochs

            batch: SequenceBatch_T
            for batch_counter, batch in enumerate(loader):
                seq_ids = batch["seq_ids"].to(device)
                attn_masks = batch["attn_masks"].to(device)
                subwords_start = batch["subwords_start"].to(device)
                idx_convertor = batch["idx_convertor"].to(device)
                heads_true = batch["heads"].to(device)
                deprels_true = batch["deprels"].to(device)
                uposs_true = batch["uposs"].to(device)
                xposs_true = batch["xposs"].to(device)
                feats_true = batch["feats"].to(device)
                lemma_scripts_true = batch["lemma_scripts"].to(device)

                model_output = self.forward(seq_ids, attn_masks)
                
                heads_pred = model_output["heads"].detach()
                deprels_pred = model_output["deprels"].detach()
                uposs_pred = model_output["uposs"].detach()
                xposs_pred = model_output["xposs"].detach()
                feats_pred = model_output["feats"].detach() 
                lemma_scripts_pred = model_output["lemma_scripts"].detach()

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
                n_correct_LAS_chuliu_batch, _, n_total_batch = \
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
                
                good_uposs_batch, total_uposs_batch = compute_acc_upos(uposs_pred, uposs_true, eps=0)
                good_uposs_epoch += good_uposs_batch
                total_uposs_epoch += total_uposs_batch

                good_xposs_batch, total_xposs_batch = compute_acc_upos(xposs_pred, xposs_true, eps=0)
                good_xposs_epoch += good_xposs_batch
                total_xposs_epoch += total_xposs_batch

                good_feats_batch, total_feats_batch = compute_acc_upos(feats_pred, feats_true, eps=0)
                good_feats_epoch += good_feats_batch
                total_feats_epoch += total_feats_batch

                good_lemma_scripts_batch, total_lemma_scripts_batch = compute_acc_upos(lemma_scripts_pred, lemma_scripts_true, eps=0)
                good_lemma_scripts_epoch += good_lemma_scripts_batch
                total_lemma_scripts_epoch += total_lemma_scripts_batch

                loss_uposs_batch = compute_loss_poss(uposs_pred, uposs_true, self.criterion)
                loss_uposs_epoch += loss_uposs_batch

                loss_xposs_batch = compute_loss_poss(xposs_pred, xposs_true, self.criterion)
                loss_xposs_epoch += loss_xposs_batch

                loss_feats_batch = compute_loss_poss(feats_pred, feats_true, self.criterion)
                loss_feats_epoch += loss_feats_batch

                loss_lemma_scripts_batch = compute_loss_poss(lemma_scripts_pred, lemma_scripts_true, self.criterion)
                loss_lemma_scripts_epoch += loss_lemma_scripts_batch

                processed_sentence_counter += seq_ids.size(0)
                time_from_start = timer() - start
                parsing_speed = int(round(((processed_sentence_counter + 1) / time_from_start) / 100, 2) * 100)
                
                if batch_counter % print_every == 0:
                    print(
                    f'Evaluating: {100 * (batch_counter + 1) / len(loader):.2f}% complete. {time_from_start:.2f} seconds in epoch ({parsing_speed:.2f} sents/sec)',
                    end='\r')


            loss_head_epoch = loss_head_epoch/len(loader)
            acc_head_epoch = good_head_epoch/total_head_epoch
            
            loss_deprel_epoch = loss_deprel_epoch/len(loader)
            acc_deprel_epoch = good_deprel_epoch/total_deprel_epoch
            
            acc_uposs_epoch = good_uposs_epoch/total_uposs_epoch
            
            acc_xposs_epoch = good_xposs_epoch/total_xposs_epoch
            
            acc_feats_epoch = good_feats_epoch/total_feats_epoch
            
            acc_lemma_scripts_epoch = good_lemma_scripts_epoch/total_lemma_scripts_epoch

            LAS_epoch = n_correct_LAS_epoch/n_total_epoch
            LAS_chuliu_epoch = n_correct_LAS_chuliu_epoch/n_total_epoch


            loss_epoch = loss_head_epoch + loss_deprel_epoch + loss_uposs_epoch + loss_xposs_epoch + loss_feats_epoch + loss_lemma_scripts_epoch
            print("\nevaluation result: LAS={:.3f}; LAS_chuliu={:.3f}; loss_epoch={:.3f}; eval_acc_head={:.3f}; eval_acc_deprel = {:.3f}, eval_acc_upos = {:.3f}, eval_acc_feat = {:.3f}, eval_acc_lemma_script = {:.3f}, acc_xposs_epoch = {:.3f}\n".format(
            LAS_epoch, LAS_chuliu_epoch, loss_epoch, LAS_epoch, acc_head_epoch, acc_uposs_epoch, acc_feats_epoch, acc_lemma_scripts_epoch, acc_xposs_epoch))

        results = {
            "LAS_epoch": LAS_epoch,
            "LAS_chuliu_epoch": LAS_chuliu_epoch,
            "acc_head_epoch": acc_head_epoch,
            "acc_deprel_epoch" : acc_deprel_epoch,
            "acc_uposs_epoch": acc_uposs_epoch,
            "acc_xposs_epoch": acc_xposs_epoch,
            "acc_feats_epoch": acc_feats_epoch,
            "acc_lemma_scripts_epoch": acc_lemma_scripts_epoch,
            "loss_head_epoch": loss_head_epoch,
            "loss_deprel_epoch": loss_deprel_epoch,
            "loss_xposs_epoch": loss_xposs_epoch,
            "loss_feats_epoch": loss_feats_epoch,
            "loss_lemma_scripts_epoch": loss_lemma_scripts_epoch,
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

    def load_pretrained(self, overwrite_pretrain_classifiers=False):
        params = self.pretrain_model_params or self.model_params
        ckpt_fpath = os.path.join(params["root_folder_path"], params["model_name"] + ".pt")
        checkpoint_state = torch.load(ckpt_fpath)
        
        tagger_pretrained_dict = self.tagger_layer.state_dict()
        for layer_name, weights in checkpoint_state["tagger"].items():
            if overwrite_pretrain_classifiers == True and layer_name in [
                "deprel.pairwise_weight", 
                "uposs_ffn.weight", 
                "uposs_ffn.bias",
                "xposs_ffn.weight",
                "xposs_ffn.bias",
                "lemma_scripts_ffn.weight",
                "lemma_scripts_ffn.bias",
                "feats_ffn.weight",
                "feats_ffn.bias",
                ]:
                print(f"Overwritting pretrained layer {layer_name}")
                continue
            tagger_pretrained_dict[layer_name] = weights
        self.tagger_layer.load_state_dict(tagger_pretrained_dict)
        
        
        llm_pretrained_dict = self.llm_layer.state_dict()
        for layer_name, weights in checkpoint_state["adapter"].items():
            if self.pretrain_model_params:
                layer_name = layer_name.replace(self.pretrain_model_params["model_name"], self.model_params["model_name"])
            if layer_name in llm_pretrained_dict:
                llm_pretrained_dict[layer_name] = weights
        self.llm_layer.load_state_dict(llm_pretrained_dict)
        return

### To reactivate if probleme in the loading of the model states
# loaded_state_dict = OrderedDict()
# for k, v in checkpoint["state_dict"].items():
#     name = k.replace("module.", "")
#     loaded_state_dict[name] = v