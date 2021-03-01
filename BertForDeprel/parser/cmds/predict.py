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
                    ) = model.forward(seq, attn_masks)
                    heads_pred, deprels_main_pred, deprels_aux_pred, poss_pred = (
                        heads_pred.detach(),
                        deprels_main_pred.detach(),
                        deprels_aux_pred.detach(),
                        poss_pred.detach(),
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

                    ###
                    # gov_dict = {}

                    # root_idx = heads_pred_np[:,1:].argmax(axis=1)[0] + 1
                    # gov_dict[root_idx] = 0
                    # r = minimum_spanning_tree(-1*heads_pred_np[1:,1:])

                    # list_done = []
                    # list_todo = [root_idx-1]
                    # gov_dict = {}
                    # gov_dict[root_idx] = 0
                    # while list_todo:
                    #     todo_idx = list_todo.pop(0)
                    #     for idx in np.where(r.toarray()[todo_idx,:] != 0)[0]:
                    #         if idx not in list_done:
                    #             list_todo.append(idx)
                    #             gov_dict[idx+1] = todo_idx+1
                    #     for idx in np.where(r.toarray()[:,todo_idx] != 0)[0]:
                    #         if idx not in list_done:
                    #             list_todo.append(idx)
                    #             gov_dict[idx+1] = todo_idx+1

                    #     list_done.append(todo_idx)

                    # heads_pred_mst = heads_pred.clone()

                    # poss_pred_list = poss_pred.max(dim=2)[1][subwords_start==1].tolist()

                    # heads_pred_mst_list = []
                    # for i in range(1, len(gov_dict)+1):
                    #     if gov_dict.get(i) == None:
                    #         print("gov_dict", gov_dict)
                    #         # print("heads_true", heads_true)
                    #         print(root_idx)
                    #         print(r)
                    #         print(heads_pred_np.argmax(axis=1))
                    #         break
                    #     heads_pred_mst_list.append(gov_dict[i])
                    # head_true_like = heads_pred.max(dim=1)[1]
                    # heads_pred_mst = head_true_like.clone().cpu().numpy()
                    # for n, sub in enumerate(subwords_start.squeeze(0).tolist()):
                    #     if (sub == 1) & (heads_pred_mst_list != []):
                    #         heads_pred_mst[:,n] = idx_convertor.squeeze(0).tolist()[heads_pred_mst_list.pop(0)]
                    #     else:
                    #         heads_pred_mst[:,n] = args.maxlen - 1
                    # heads_pred_mst = torch.tensor(heads_pred_mst).to(args.device)

                    # deprels_main_pred_mst = deprel_aligner_with_head(deprels_main_pred, heads_pred_mst)
                    # deprels_main_pred_mst_list = deprels_main_pred_mst.max(dim=1)[1][subwords_start==1].tolist()

                    # deprels_aux_pred_mst = deprel_aligner_with_head(deprels_aux_pred, heads_pred_mst)
                    # deprels_aux_pred_mst_list = deprels_aux_pred_mst.max(dim=1)[1][subwords_start==1].tolist()
                    ###

                    idx2head = []
                    sum_idx = 0
                    for idx, sub in enumerate(subwords_start.squeeze(0).tolist()):
                        sum_idx += max(0, sub)
                        idx2head.append(sum_idx)
                    # print("idx2head", idx2head)
                    ### To delete first one
                    # conllu_sequence = pred_dataset.dataset.sequences[pred_dataset.indices[n_sentence]]
                    conllu_sequence = pred_dataset.sequences[n_sentence]

                    # conllu_sequence = pred_dataset.sequences[n_sentence]
                    poped_item = 0
                    for n_token in range(len(conllu_sequence)):

                        if type(conllu_sequence[n_token - poped_item]["id"]) != int:
                            print("POP :", token)
                            conllu_sequence.pop(n_token - poped_item)
                            poped_item += 1
                    for n_token, (pos_index, head_chuliu, dmpmstn, dapmst) in enumerate(
                        zip(
                            poss_pred_list,
                            chuliu_heads_list,
                            deprels_main_pred_chuliu_list,
                            deprels_aux_pred_chuliu_list,
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
                            token["misc"] = misc


                        else:
                            # token["head"] = gov_dict.get(n_token+1, 'missing_gov')
                            token["head"] = str(head_chuliu)
                            token["upostag"] = args.i2pos[pos_index]
                            if loaded_args.split_deprel:

                                if args.i2dra[dapmst] == "none":
                                    token["deprel"] = args.i2drm[dmpmstn]
                                else:
                                    token["deprel"] = "{}:{}".format(
                                        args.i2drm[dmpmstn], args.i2dra[dapmst]
                                    )
                            else:
                                token["deprel"] = args.i2drm[dmpmstn]

                        ### MISC PART
                        # misc = token["misc"]
                        # if not misc:
                        #     misc = OrderedDict()
                        # # misc["head_pred"] = str(idx2head[head_pred])
                        # # misc["head_MST"] = str(gov_dict.get(n_token+1, 'NA'))
                        # # print(n_token + 1, misc["head_MST"])
                        # misc["upostag_pred"] = args.i2pos[pos_index]
                        # misc["heads_mst"] = str(gov_dict.get(n_token+1, 'missing_gov'))
                        # # misc["deprel_main_pred"] = args.i2drm[dmp]
                        # # misc["deprel_aux_pred"] = args.i2dra[dap]
                        # token["misc"] = misc
                        # # ### END MISC PART

                        # print(n_token, conllu_sequence[n_token]["form"], head_pred, head_true, args.i2drm[dmt], args.i2drm[dmp], args.i2dra[dap], args.i2dra[dat])
                    # print(conllu_sequence.serialize())
                    list_conllu_sequences.append(conllu_sequence)
                    # print("")
                    print(
                        f"Predicting: {100 * (n_sentence + 1) / len(pred_dataset):.2f}% complete. {timer() - start:.2f} seconds in epoch.",
                        end="\r",
                    )

            with open(path_result_file, "w") as f:
                f.writelines(
                    [sequence.serialize() for sequence in list_conllu_sequences]
                )
