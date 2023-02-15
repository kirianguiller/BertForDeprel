import os
from collections import OrderedDict
from timeit import default_timer as timer

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..cmds.cmd import CMD
from ..utils.lemma_script_utils import apply_lemma_rule
from ..utils.chuliu_edmonds_utils import chuliu_edmonds_one_root
from ..utils.load_data_utils import ConlluDataset
from ..utils.model_utils import BertForDeprel
from ..utils.scores_and_losses_utils import deprel_aligner_with_head
from ..utils.types import ModelParams_T

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
        subparser.add_argument("--batch_size", default=4, type=int, help="batch size")
        subparser.add_argument("--infile", '-i', required=True, help="path to infile (can be a folder)")
        subparser.add_argument("--outfile", '-o',help="path to predicted outfile(s)")
        subparser.add_argument(
            "--overwrite", action="store_true", help="whether to overwrite predicted file if already existing"
        )
        subparser.add_argument(
            "--write_preds_in_misc",
            action="store_true",
            help="whether to include punctuation",
        )

        return subparser

    def __call__(self, args, model_params: ModelParams_T):
        super(Predict, self).__call__(args, model_params)
        if not args.conf:
            raise Exception("Path to model xxx.config.json must be provided as --conf parameter")
        paths_pred = []
        if os.path.isdir(args.infile):
            for file in os.listdir(args.infile):
                paths_pred.append(os.path.join(args.infile, file))
        elif os.path.isfile(args.infile):
            paths_pred.append(args.infile)
        else:
            raise BaseException(f"args.infile must be a folder or a file, not nothing (current infile = {args.infile})")
        
        path_predicted_files = args.outfile

        if not os.path.isdir(path_predicted_files):
            os.makedirs(path_predicted_files)

        print(paths_pred)

        print("Load the model")
        model = BertForDeprel(model_params)
        model.load_pretrained()


        model.to(args.device)

        if args.multi_gpu:
            print("MODEL TO MULTI GPU")
            model = nn.DataParallel(model)

        tokenizer = AutoTokenizer.from_pretrained(model_params["embedding_type"])

        model.eval()
        print("Starting Predictions ...")
        for path in paths_pred:

            path_result_file = os.path.join(path_predicted_files, path.split("/")[-1])
            
            if args.overwrite != True:
                if os.path.isfile(path_result_file):
                    print(f"file '{path_result_file}' already exist and overwrite!=False, skipping ...\n")
                    continue
                
            print(args.infile)

            print("Load the dataset")
            
            pred_dataset = ConlluDataset(path, tokenizer, model_params, args.mode)

            params = {
                # "batch_size": args.batch_size,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            }
            annotation_schema = model_params["annotation_schema"]
            pred_loader = DataLoader(pred_dataset, collate_fn=pred_dataset.collate_fn, shuffle=False, **params)
            print(
                f"{'eval:':6} {len(pred_dataset):5} sentences, "
                f"{len(pred_loader):3} batches, "
            )
            start = timer()
            list_conllu_sequences = []
            with torch.no_grad():
                for batch_counter, (
                    seq_batch,
                    subwords_start_batch,
                    attn_masks_batch,
                    idx_convertor_batch,
                    sentence_idxs_batch,
                ) in enumerate(pred_loader):

                    seq_batch, attn_masks_batch = seq_batch.to(args.device), attn_masks_batch.to(args.device)
                    (
                        heads_pred_batch,
                        deprels_main_pred_batch,
                        poss_pred_batch,
                    ) = model.forward(seq_batch, attn_masks_batch)
                    heads_pred_batch, deprels_main_pred_batch, poss_pred_batch = (
                        heads_pred_batch.detach(),
                        deprels_main_pred_batch.detach(),
                        poss_pred_batch.detach(),
                    )

                    for sentence_in_batch_counter in range(seq_batch.size()[0]):
                        # sentence_idxs_batch

                        subwords_start = subwords_start_batch[sentence_in_batch_counter]
                        idx_convertor = idx_convertor_batch[sentence_in_batch_counter]
                        heads_pred = heads_pred_batch[sentence_in_batch_counter].clone()
                        deprels_main_pred = deprels_main_pred_batch[sentence_in_batch_counter].clone()
                        poss_pred = poss_pred_batch[sentence_in_batch_counter].clone()
                        sentence_idx = sentence_idxs_batch[sentence_in_batch_counter]
                        
                        n_sentence = int(sentence_idx)
                        
                        head_true_like = heads_pred.max(dim=0).indices
                        chuliu_heads_pred = head_true_like.clone().cpu().numpy()
                        chuliu_heads_list = []

                        subwords_start_with_root = subwords_start.clone()
                        subwords_start_with_root[0] = True

                        heads_pred_np = heads_pred[
                            :, subwords_start_with_root == 1
                        ][subwords_start_with_root == 1]

                        heads_pred_np = heads_pred_np.cpu().numpy()

                        chuliu_heads_vector = chuliu_edmonds_one_root(
                            np.transpose(heads_pred_np, (1, 0))
                        )[1:]
                        for i_token, chuliu_head_pred in enumerate(chuliu_heads_vector):
                            chuliu_heads_pred[
                                idx_convertor[i_token + 1]
                            ] = idx_convertor[chuliu_head_pred]
                            chuliu_heads_list.append(chuliu_head_pred)

                        chuliu_heads_pred = torch.tensor(chuliu_heads_pred).to(args.device)

                        deprels_main_pred_chuliu = deprel_aligner_with_head(
                            deprels_main_pred.unsqueeze(0), chuliu_heads_pred.unsqueeze(0)
                        ).squeeze(0)
                        
                        deprels_main_pred_chuliu_list = deprels_main_pred_chuliu.max(dim=0).indices[subwords_start == 1].tolist()

                        poss_pred_list = poss_pred.max(dim=1).indices[
                            subwords_start == 1
                        ].tolist()

                        # lemma_scripts_pred_list = lemma_scripts_pred.max(dim=2)[1][
                        #     subwords_start == 1
                        # ].tolist()

                        idx2head = []
                        sum_idx = 0
                        for sub in subwords_start.tolist():
                            sum_idx += max(0, sub)
                            idx2head.append(sum_idx)
                        conllu_sequence = pred_dataset.sequences[n_sentence]

                        poped_item = 0
                        for n_token in range(len(conllu_sequence)):

                            if type(conllu_sequence[n_token - poped_item]["id"]) != int:
                                print("POP :", token)
                                conllu_sequence.pop(n_token - poped_item)
                                poped_item += 1
                        for n_token, (pos_index, head_chuliu, dmpmstn) in enumerate(
                            zip(
                                poss_pred_list,
                                chuliu_heads_list,
                                deprels_main_pred_chuliu_list,
                                # lemma_scripts_pred_list,
                            )
                        ):
                            token = conllu_sequence[n_token]

                            if args.write_preds_in_misc:
                                misc = token["misc"]
                                if not misc:
                                    misc = OrderedDict()
                                misc["deprel_main_pred"] = annotation_schema["deprels"][dmpmstn]

                                # misc['head_MST']= str(gov_dict.get(n_token+1, 'missing_gov'))
                                misc["head_MST_pred"] = str(head_chuliu)
                                misc["upostag_pred"] = annotation_schema["uposs"][pos_index]
                                # lemma_script = annotation_schema["i2lemma_script"][lemma_script_index]
                                # misc["lemma_pred"] = apply_lemma_rule(token["form"], lemma_script)
                                token["misc"] = misc


                            else:
                                # token["head"] = gov_dict.get(n_token+1, 'missing_gov')
                                token["head"] = str(head_chuliu)
                                token["upos"] = annotation_schema["uposs"][pos_index]
                                # lemma_script = annotation_schema["i2lemma_script"][lemma_script_index]
                                # token["lemma"] = apply_lemma_rule(token["form"], lemma_script)
                                token["deprel"] = annotation_schema["deprels"][dmpmstn]


                        list_conllu_sequences.append(conllu_sequence)
                        time_from_start = timer() - start
                        parsing_speed = int(round(((n_sentence + 1) / time_from_start) / 100, 2) * 100)
                        print(
                            f"Predicting: {100 * (n_sentence + 1) / len(pred_dataset):.2f}% complete. {time_from_start:.2f} seconds in epoch ({parsing_speed} sents/sec).",
                            end="\r",
                        )

            with open(path_result_file, "w") as f:
                f.writelines(
                    [sequence.serialize() for sequence in list_conllu_sequences]
                )
            
            print(f"Finished predicting `{path_result_file}, wrote {n_sentence + 1} sents in {round(timer() - start, 2)} secs`")
        
        
