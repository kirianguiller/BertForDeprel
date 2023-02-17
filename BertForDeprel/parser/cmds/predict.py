import os
from typing import List
from conllup.conllup import sentenceJsonToConll
from timeit import default_timer as timer

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from torch import nn
from torch.utils.data import DataLoader

from ..cmds.cmd import CMD
from ..utils.annotation_schema_utils import get_path_of_conllus_from_folder_path
from ..utils.chuliu_edmonds_utils import chuliu_edmonds_one_root
from ..utils.load_data_utils import ConlluDataset, SequenceBatch_T
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
        subparser.add_argument("--inpath", '-i', required=True, help="path to inpath (can be a folder)")
        subparser.add_argument("--outpath", '-o',help="path to predicted outpath(s)")
        subparser.add_argument("--suffix", default="", help="suffix that will be added to the name of the predicted files (before the file extension)")
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
        if os.path.isdir(args.inpath):
            paths_pred = get_path_of_conllus_from_folder_path(args.inpath)
        elif os.path.isfile(args.inpath):
            paths_pred.append(args.inpath)
        else:
            raise BaseException(f"args.inpath must be a folder or a file, not nothing (current inpath = {args.inpath})")
        
        path_predicted_files = args.outpath

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

        model.eval()
        print("Starting Predictions ...")
        for path in paths_pred:

            path_result_file = os.path.join(path_predicted_files, path.split("/")[-1].replace(".conll", args.suffix + ".conll"))
            
            if args.overwrite != True:
                if os.path.isfile(path_result_file):
                    print(f"file '{path_result_file}' already exist and overwrite!=False, skipping ...\n")
                    continue
                
            print(args.inpath)

            print("Load the dataset")
            
            pred_dataset = ConlluDataset(path, model_params, args.mode)

            params = {
                "batch_size": model_params["batch_size"],
                "num_workers": args.num_workers,
            }
            pred_loader = DataLoader(pred_dataset, collate_fn=pred_dataset.collate_fn, shuffle=False, **params)
            print(
                f"{'eval:':6} {len(pred_dataset):5} sentences, "
                f"{len(pred_loader):3} batches, "
            )
            start = timer()
            list_conllu_sequences: List[str] = []
            batch: SequenceBatch_T
            with torch.no_grad():
                for batch in pred_loader:
                    seq_ids_batch = batch["seq_ids"].to(args.device)
                    attn_masks_batch = batch["attn_masks"].to(args.device)
                    subwords_start_batch = batch["subwords_start"]
                    idx_convertor_batch = batch["idx_convertor"]
                    idx_batch = batch["idx"]

                    preds = model.forward(seq_ids_batch, attn_masks_batch)
                    heads_pred_batch = preds["heads"].detach()
                    deprels_pred_batch = preds["deprels"].detach()
                    uposs_pred_batch = preds["uposs"].detach()
                    xposs_pred_batch = preds["xposs"].detach()
                    feats_pred_batch = preds["feats"].detach()
                    lemma_scripts_pred_batch = preds["lemma_scripts"].detach()

                    for sentence_in_batch_counter in range(seq_ids_batch.size()[0]):
                        subwords_start = subwords_start_batch[sentence_in_batch_counter]
                        idx_convertor = idx_convertor_batch[sentence_in_batch_counter]
                        heads_pred = heads_pred_batch[sentence_in_batch_counter].clone()
                        deprels_pred = deprels_pred_batch[sentence_in_batch_counter].clone()
                        uposs_pred = uposs_pred_batch[sentence_in_batch_counter].clone()
                        xposs_pred = xposs_pred_batch[sentence_in_batch_counter].clone()
                        feats_pred = feats_pred_batch[sentence_in_batch_counter].clone()
                        lemma_scripts_pred = lemma_scripts_pred_batch[sentence_in_batch_counter].clone()
                        sentence_idx = idx_batch[sentence_in_batch_counter]
                        
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
                            chuliu_heads_list.append(int(chuliu_head_pred))

                        chuliu_heads_pred = torch.tensor(chuliu_heads_pred).to(args.device)

                        deprels_pred_chuliu = deprel_aligner_with_head(
                            deprels_pred.unsqueeze(0), chuliu_heads_pred.unsqueeze(0)
                        ).squeeze(0)
                        
                        deprels_pred_chuliu_list = deprels_pred_chuliu.max(dim=0).indices[subwords_start == 1].tolist()

                        uposs_pred_list = uposs_pred.max(dim=1).indices[
                            subwords_start == 1
                        ].tolist()

                        xposs_pred_list = xposs_pred.max(dim=1).indices[
                            subwords_start == 1
                        ].tolist()

                        feats_pred_list = feats_pred.max(dim=1).indices[
                            subwords_start == 1
                        ].tolist()

                        lemma_scripts_pred_list = lemma_scripts_pred.max(dim=1).indices[
                            subwords_start == 1
                        ].tolist()


                        predicted_sentence_json = pred_dataset.add_prediction_to_sentence_json(
                            n_sentence,
                            uposs_pred_list,
                            xposs_pred_list,
                            chuliu_heads_list,
                            deprels_pred_chuliu_list,
                            feats_pred_list,
                            lemma_scripts_pred_list,
                        )
                        list_conllu_sequences.append(sentenceJsonToConll(predicted_sentence_json))

                        time_from_start = timer() - start
                        parsing_speed = int(round(((n_sentence + 1) / time_from_start) / 100, 2) * 100)
                        print(
                            f"Predicting: {100 * (n_sentence + 1) / len(pred_dataset):.2f}% complete. {time_from_start:.2f} seconds in epoch ({parsing_speed} sents/sec).",
                            end="\r",
                        )

            with open(path_result_file, "w") as f:
                f.write("\n\n".join(list_conllu_sequences))
            
            print(f"Finished predicting `{path_result_file}, wrote {n_sentence + 1} sents in {round(timer() - start, 2)} secs`")
        
        
