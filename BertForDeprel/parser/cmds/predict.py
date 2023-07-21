from argparse import ArgumentParser
import os
from typing import List, Tuple
from conllup.conllup import writeConlluFile, sentenceJson_T
from timeit import default_timer as timer

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree # type: ignore (TODO: why can't PyLance find this?)
from torch import Tensor, nn
from torch.utils.data import DataLoader


from parser.modules.BertForDepRelOutput import BertForDeprelBatchOutput
from ..cmds.cmd import CMD, SubparsersType
from ..utils.annotation_schema_utils import get_path_of_conllus_from_folder_path
from ..utils.chuliu_edmonds_utils import chuliu_edmonds_one_root_with_constraints
from ..utils.load_data_utils import ConlluDataset, CopyOption, PartialPredictionConfig, SequencePredictionBatch_T
from ..modules.BertForDepRel import BertForDeprel
from ..utils.scores_and_losses_utils import deprel_aligner_with_head
from ..utils.types import ModelParams_T


def max_span_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix_inverted = -1 * matrix
    max_span_inverted = minimum_spanning_tree(matrix_inverted)
    return -1 * (max_span_inverted.toarray().astype(int))


class Predict(CMD):
    def add_subparser(self, name: str, parser: SubparsersType) -> ArgumentParser:
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
        subparser.add_argument(
            "--keep_heads", default="NONE",
            help="whether to use deps of input files as constrained for maximum spanning tree (NONE | EXISTING | ALL) (default : NONE)",
        )
        subparser.add_argument(
            "--keep_deprels", default="NONE", help="whether to keep current deprels and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )
        subparser.add_argument(
            "--keep_upos", default="NONE", help="whether to keep current upos and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )
        subparser.add_argument(
            "--keep_xpos", default="NONE", help="whether to keep current xpos and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )
        subparser.add_argument(
            "--keep_feats", default="NONE", help="whether to keep current feats and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )
        subparser.add_argument(
            "--keep_lemmas", default="NONE", help="whether to keep current lemmas and not predict new ones (NONE | EXISTING | ALL) (default : NONE)"
        )

        return subparser

    # TODO: there must be some overlap between this and the model eval code
    def __get_constrained_dependencies(self, heads_pred, deprels_pred, subwords_start, keep_heads: CopyOption, pred_dataset: ConlluDataset, n_sentence: int, idx_converter: Tensor, device: str):
        head_true_like = heads_pred.max(dim=0).indices
        chuliu_heads_pred = head_true_like.clone().cpu().numpy()
        chuliu_heads_list: List[int] = []

        subwords_start_with_root = subwords_start.clone()
        subwords_start_with_root[0] = True
        heads_pred_np = heads_pred[
            :, subwords_start_with_root == 1
        ][subwords_start_with_root == 1]

        heads_pred_np = heads_pred_np.cpu().numpy()

        forced_relations: List[Tuple] = []
        if keep_heads == "EXISTING":
            forced_relations = pred_dataset.get_constrained_dependency_for_chuliu(n_sentence)

        chuliu_heads_vector = chuliu_edmonds_one_root_with_constraints(
            np.transpose(heads_pred_np, (1, 0)), forced_relations
        )[1:]
        for i_dependent_word, chuliu_head_pred in enumerate(chuliu_heads_vector):
            chuliu_heads_pred[
                idx_converter[i_dependent_word + 1]
            ] = idx_converter[chuliu_head_pred]
            chuliu_heads_list.append(int(chuliu_head_pred))

        chuliu_heads_pred = torch.tensor(chuliu_heads_pred).to(device)

        deprels_pred_chuliu = deprel_aligner_with_head(
            deprels_pred.unsqueeze(0), chuliu_heads_pred.unsqueeze(0)
        ).squeeze(0)

        return chuliu_heads_list, deprels_pred_chuliu

    def __prediction_iterator(self, batch: SequencePredictionBatch_T, preds: BertForDeprelBatchOutput, pred_dataset: ConlluDataset, partial_pred_config: PartialPredictionConfig, device: str):
        subwords_start_batch = batch.subwords_start
        idx_converter_batch = batch.idx_converter
        idx_batch = batch.idx

        for sentence_in_batch_counter in range(batch.sequence_token_ids.size()[0]):
            # Next: push these two into the output classes so that the code farther below can be encapsulated in the output classes;
            # these will then be containers for the raw model outputs, with methods for constructing the final predictions.
            # the overwrite logic should be done in a separate step, I think.
            subwords_start = subwords_start_batch[sentence_in_batch_counter]
            idx_converter = idx_converter_batch[sentence_in_batch_counter]
            raw_sentence_preds = preds.distributions_for_sentence(sentence_in_batch_counter)

            sentence_idx = idx_batch[sentence_in_batch_counter]
            n_sentence = int(sentence_idx)

            (chuliu_heads_list, deprels_pred_chuliu) = self.__get_constrained_dependencies(
                heads_pred=raw_sentence_preds.heads,
                deprels_pred=raw_sentence_preds.deprels,
                subwords_start=subwords_start,
                pred_dataset=pred_dataset,
                keep_heads=partial_pred_config.keep_heads,
                n_sentence=n_sentence,
                idx_converter=idx_converter,
                device=device,)

            # TODO: would it not be better if this were a boolean vector to begin with?
            is_word_start = subwords_start == 1

            deprels_pred_chuliu_list = deprels_pred_chuliu.max(dim=0).indices[
                is_word_start
            ].tolist()

            uposs_pred_list = raw_sentence_preds.uposs.max(dim=1).indices[
                is_word_start
            ].tolist()

            xposs_pred_list = raw_sentence_preds.xposs.max(dim=1).indices[
                is_word_start
            ].tolist()

            feats_pred_list = raw_sentence_preds.feats.max(dim=1).indices[
                is_word_start
            ].tolist()

            lemma_scripts_pred_list = raw_sentence_preds.lemma_scripts.max(dim=1).indices[
                is_word_start
            ].tolist()

            yield pred_dataset.construct_sentence_prediction(
                            n_sentence,
                            uposs_pred_list,
                            xposs_pred_list,
                            chuliu_heads_list,
                            deprels_pred_chuliu_list,
                            feats_pred_list,
                            lemma_scripts_pred_list,
                            partial_pred_config=partial_pred_config
                        )


    def __call__(self, args, model_params: ModelParams_T):
        super().__call__(args, model_params)
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

        partial_pred_config = PartialPredictionConfig(
            keep_upos=args.keep_upos,
            keep_xpos=args.keep_xpos,
            keep_feats=args.keep_feats,
            keep_deprels=args.keep_deprels,
            keep_heads=args.keep_heads,
            keep_lemmas=args.keep_lemmas
            )

        print(paths_pred)

        print("Loading model...")
        model = BertForDeprel(model_params)
        model.load_pretrained()


        model.to(args.device)

        if args.multi_gpu:
            print("Sending model to multiple GPUs...")
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

            print(f"Loading dataset from {args.inpath}...")

            pred_dataset = ConlluDataset(path, model_params, args.mode)

            params = {
                "batch_size": model_params.batch_size,
                "num_workers": args.num_workers,
            }
            pred_loader = DataLoader(pred_dataset, collate_fn=pred_dataset.collate_fn_predict, shuffle=False, **params)
            print(
                f"{'Loaded '} {len(pred_dataset):5} sentences ({len(pred_loader):3} batches)"
            )
            start = timer()
            predicted_sentences: List[sentenceJson_T] = []
            parsed_sentence_counter = 0
            batch: SequencePredictionBatch_T
            with torch.no_grad():
                for batch in pred_loader:
                    batch = batch.to(args.device)
                    preds = model.forward(batch.sequence_token_ids, batch.attn_masks).detach()

                    time_from_start = 0
                    parsing_speed = 0
                    for predicted_sentence in self.__prediction_iterator(batch, preds, pred_dataset, partial_pred_config, args.device):
                        predicted_sentences.append(predicted_sentence)

                        parsed_sentence_counter += 1
                        time_from_start = timer() - start
                        parsing_speed = int(round(((parsed_sentence_counter + 1) / time_from_start) / 100, 2) * 100)

                    print(
                        f"Predicting: {100 * (parsed_sentence_counter + 1) / len(pred_dataset):.2f}% complete. {time_from_start:.2f} seconds in file ({parsing_speed} sents/sec).",
                        end="\r",
                    )

            writeConlluFile(path_result_file, predicted_sentences, overwrite=args.overwrite)

            print(f"Finished predicting `{path_result_file}, wrote {parsed_sentence_counter} sents in {round(timer() - start, 2)} secs`")


