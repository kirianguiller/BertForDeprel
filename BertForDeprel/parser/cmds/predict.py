# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import softmax
from torch import nn
from torch.utils.data import DataLoader, random_split


from ..cmds.cmd import CMD
from ..utils.lemma_script_utils import apply_lemma_rule
from ..utils.chuliu_edmonds_utils import chuliu_edmonds_one_root
from ..utils.load_data_utils import ConlluDataset
from ..utils.model_utils import BertForDeprel
from ..utils.os_utils import path_or_name
from ..utils.train_utils import deprel_aligner_with_head, eisner, eval_epoch


def min_span_matrix(matrix):
    matrix = -1 * matrix
    return minimum_spanning_tree(matrix)


# np.ndarray -> np.ndarray
def max_span_matrix(matrix):
    m = min_span_matrix(matrix)
    return -1 * (m.toarray().astype(int))


class Predict(CMD):
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help="Use a trained model to make predictions."
        )
        subparser.add_argument("--batch_size", default=1, type=int, help="batch size")
        subparser.add_argument(
            "--punct", action="store_true", help="whether to include punctuation"
        )
        subparser.add_argument("--fpred", default="", help="path to dataset")
        subparser.add_argument(
            "--upostag", action="store_true", help="whether to predict POS"
        )
        subparser.add_argument(
            "--multiple", action="store_true", help="whether to include punctuation"
        )
        subparser.add_argument(
            "--overwrite", action="store_true", help="whether to overwrite predicted file if already existing"
        )
        subparser.add_argument(
            "--write_preds_in_misc",
            action="store_true",
            help="whether to include punctuation",
        )
        # subparser.add_argument('--fresults', default='',
        #                        help='path to predicted result')

        return subparser

    def __call__(self, args):
        super(Predict, self).__call__(args)
        if not args.fpred:
            args.fpred = os.path.join(args.folder, "to_predict")
        
        paths_pred = []
        if os.path.isdir(args.fpred):
            for file in os.listdir(args.fpred):
                paths_pred.append(os.path.join(args.fpred, file))
        elif os.path.isfile(args.fpred):
            paths_pred.append(args.fpred)
        else:
            raise BaseException(f"args.fpred must be a folder or a file, not nothing (current fpred = {args.fpred})")
        
        path_predicted_files = os.path.join(args.folder, "predicted")
        if not os.path.isdir(path_predicted_files):
            os.makedirs(path_predicted_files)

        print(paths_pred)

        if not os.path.isdir(os.path.join(args.folder, "results")):
            os.makedirs(os.path.join(args.folder, "results"))

        print("Load the saved config")
        checkpoint = torch.load(args.model, map_location=torch.device("cpu"))
        loaded_args = checkpoint["args"]
        loaded_args.mode = "predict"
        args.bert_type = loaded_args.bert_type
        print(args.num_workers)
        print(type(args.num_workers))

        print("Load the model")
        model = BertForDeprel(loaded_args)

        ### To reactivate if probleme in the loading of the model states
        loaded_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            name = k.replace("module.", "")
            loaded_state_dict[name] = v

        model.load_state_dict(loaded_state_dict)
        model.to(args.device)

        if args.multi_gpu:
            print("MODEL TO MULTI GPU")
            model = nn.DataParallel(model)

        
        self.load_tokenizer(loaded_args.bert_type)
        # model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print("Starting Predictions ...")
        for path in paths_pred:

            path_result_file = os.path.join(path_predicted_files, path.split("/")[-1])
            
            if args.overwrite != True:
                if os.path.isfile(path_result_file):
                    print(f"file '{path_result_file}' already exist and overwrite!=False, skipping ...\n")
                    continue
                
            args.fpred = path
            print(args.fpred)

            print("Load the dataset")
            
            pred_dataset = ConlluDataset(args.fpred, self.tokenizer, loaded_args)

            args.drm2i = pred_dataset.drm2i
            args.i2drm = pred_dataset.i2drm
            if loaded_args.split_deprel:
                args.dra2i = pred_dataset.dra2i
                args.i2dra = pred_dataset.i2dra
            args.i2pos = pred_dataset.i2pos
            args.i2lemma_script = pred_dataset.i2lemma_script
            params = {
                # "batch_size": args.batch_size,
                "batch_size": 1,
                "num_workers": args.num_workers,
            }

            pred_loader = DataLoader(pred_dataset, **params)
            print(
                f"{'eval:':6} {len(pred_dataset):5} sentences, "
                f"{len(pred_loader):3} batches, "
            )
            start = timer()
            list_conllu_sequences = []
            with torch.no_grad():
                for n_sentence, (
                    seq,
                    subwords_start,
                    attn_masks,
                    idx_convertor,
                ) in enumerate(pred_dataset):

                    seq, attn_masks = seq.to(args.device), attn_masks.to(args.device)
                    seq = seq.unsqueeze(0)
                    attn_masks = attn_masks.unsqueeze(0)
                    subwords_start = subwords_start.unsqueeze(0)
                    idx_convertor = idx_convertor.unsqueeze(0)

                    (
                        heads_pred,
                        deprels_main_pred,
                        deprels_aux_pred,
                        poss_pred,
                        lemma_scripts_pred,
                    ) = model.forward(seq, attn_masks)
                    heads_pred, deprels_main_pred, deprels_aux_pred, poss_pred, lemma_scripts_pred = (
                        heads_pred.detach(),
                        deprels_main_pred.detach(),
                        deprels_aux_pred.detach(),
                        poss_pred.detach(),
                        lemma_scripts_pred.detach(),
                    )

                    subwords_start_with_root = subwords_start.clone()
                    subwords_start_with_root[0, 0] = True

                    heads_pred_np = heads_pred.squeeze(0)[
                        subwords_start_with_root.squeeze(0) == 1, :
                    ]
                    heads_pred_np = heads_pred_np.squeeze(0)[
                        :, subwords_start_with_root.squeeze(0) == 1
                    ]
                    heads_pred_np = heads_pred_np.cpu().numpy()

                    head_true_like = heads_pred.max(dim=1)[1]
                    chuliu_heads_pred = head_true_like.clone().cpu().numpy()
                    # chuliu_heads_pred = subwords_start.clone()
                    chuliu_heads_list = []
                    for i_vector, (
                        heads_pred_vector,
                        subwords_start_vector,
                        idx_convertor_vector,
                    ) in enumerate(zip(heads_pred, subwords_start, idx_convertor)):
                        subwords_start_with_root = subwords_start_vector.clone()
                        subwords_start_with_root[0] = True

                        heads_pred_np = heads_pred_vector[
                            :, subwords_start_with_root == 1
                        ][subwords_start_with_root == 1]
                        heads_pred_np = heads_pred_np.cpu().numpy()

                        chuliu_heads_vector = chuliu_edmonds_one_root(
                            np.transpose(heads_pred_np, (1, 0))
                        )[1:]
                        for i_token, chuliu_head_pred in enumerate(chuliu_heads_vector):
                            chuliu_heads_pred[
                                i_vector, idx_convertor_vector[i_token + 1]
                            ] = idx_convertor_vector[chuliu_head_pred]
                            chuliu_heads_list.append(chuliu_head_pred)

                    chuliu_heads_pred = torch.tensor(chuliu_heads_pred).to(args.device)

                    deprels_main_pred_chuliu = deprel_aligner_with_head(
                        deprels_main_pred, chuliu_heads_pred
                    )
                    deprels_main_pred_chuliu_list = deprels_main_pred_chuliu.max(dim=1)[
                        1
                    ][subwords_start == 1].tolist()

                    deprels_aux_pred_chuliu = deprel_aligner_with_head(
                        deprels_aux_pred, chuliu_heads_pred
                    )
                    deprels_aux_pred_chuliu_list = deprels_aux_pred_chuliu.max(dim=1)[
                        1
                    ][subwords_start == 1].tolist()

                    # print(deprels_main_pred_chuliu_list)

                    poss_pred_list = poss_pred.max(dim=2)[1][
                        subwords_start == 1
                    ].tolist()

                    lemma_scripts_pred_list = lemma_scripts_pred.max(dim=2)[1][
                        subwords_start == 1
                    ].tolist()


                    idx2head = []
                    sum_idx = 0
                    for idx, sub in enumerate(subwords_start.squeeze(0).tolist()):
                        sum_idx += max(0, sub)
                        idx2head.append(sum_idx)
                    conllu_sequence = pred_dataset.sequences[n_sentence]

                    poped_item = 0
                    for n_token in range(len(conllu_sequence)):

                        if type(conllu_sequence[n_token - poped_item]["id"]) != int:
                            print("POP :", token)
                            conllu_sequence.pop(n_token - poped_item)
                            poped_item += 1
                    for n_token, (pos_index, head_chuliu, dmpmstn, dapmst, lemma_script_index) in enumerate(
                        zip(
                            poss_pred_list,
                            chuliu_heads_list,
                            deprels_main_pred_chuliu_list,
                            deprels_aux_pred_chuliu_list,
                            lemma_scripts_pred_list,
                        )
                    ):
                        token = conllu_sequence[n_token]

                        if args.write_preds_in_misc:
                            misc = token["misc"]
                            if not misc:
                                misc = OrderedDict()
                            if loaded_args.split_deprel:
                                misc["deprel_main_pred"] = args.i2drm[dmpmstn]
                                misc["deprel_aux_pred"] = args.i2dra[dapmst]
                            else:
                                misc["deprel_main_pred"] = args.i2drm[dmpmstn]

                            # misc['head_MST']= str(gov_dict.get(n_token+1, 'missing_gov'))
                            misc["head_MST_pred"] = str(head_chuliu)
                            misc["upostag_pred"] = args.i2pos[pos_index]
                            lemma_script = args.i2lemma_script[lemma_script_index]
                            misc["lemma_pred"] = apply_lemma_rule(token["form"], lemma_script)
                            token["misc"] = misc


                        else:
                            # token["head"] = gov_dict.get(n_token+1, 'missing_gov')
                            token["head"] = str(head_chuliu)
                            token["upos"] = args.i2pos[pos_index]
                            lemma_script = args.i2lemma_script[lemma_script_index]
                            token["lemma"] = apply_lemma_rule(token["form"], lemma_script)
                            if loaded_args.split_deprel:

                                if args.i2dra[dapmst] == "none":
                                    token["deprel"] = args.i2drm[dmpmstn]
                                else:
                                    token["deprel"] = "{}:{}".format(
                                        args.i2drm[dmpmstn], args.i2dra[dapmst]
                                    )
                            else:
                                token["deprel"] = args.i2drm[dmpmstn]


                    list_conllu_sequences.append(conllu_sequence)
                    print(
                        f"Predicting: {100 * (n_sentence + 1) / len(pred_dataset):.2f}% complete. {timer() - start:.2f} seconds in epoch.",
                        end="\r",
                    )

            with open(path_result_file, "w") as f:
                f.writelines(
                    [sequence.serialize() for sequence in list_conllu_sequences]
                )
