from argparse import ArgumentParser, Namespace
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Tuple

import numpy as np
import torch
from conllup.conllup import sentenceJson_T, writeConlluFile
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..cmds.cmd import CMD, SubparsersType
from ..modules.BertForDepRel import BertForDeprel
from ..modules.BertForDepRelOutput import BertForDeprelBatchOutput
from ..utils.chuliu_edmonds_utils import chuliu_edmonds_one_root_with_constraints
from ..utils.load_data_utils import (
    CopyOption,
    PartialPredictionConfig,
    SequencePredictionBatch_T,
    UDDataset,
    load_conllu_sentences,
    resolve_conllu_paths,
)
from ..utils.scores_and_losses_utils import _deprel_pred_for_heads
from ..utils.types import ModelParams_T, PredictionConfig


class PredictCmd(CMD):
    def add_subparser(self, name: str, parser: SubparsersType) -> ArgumentParser:
        subparser = parser.add_parser(
            name, help="Use a trained model to make predictions."
        )
        subparser.add_argument(
            "--model_path", "-f", type=Path, help="Path to model to predict with"
        )
        subparser.add_argument(
            "--inpath",
            "-i",
            required=True,
            type=Path,
            help="path to inpath (can be a folder)",
        )
        subparser.add_argument(
            "--outpath", "-o", type=Path, help="path to predicted output path(s)"
        )
        subparser.add_argument(
            "--suffix",
            default="",
            help="suffix that will be added to the name of the predicted files (before"
            " the file extension)",
        )
        subparser.add_argument(
            "--overwrite",
            action="store_true",
            help="whether to overwrite predicted file if already existing",
        )
        subparser.add_argument(
            "--keep_heads",
            default="NONE",
            help="whether to use deps of input files as constrained for maximum "
            "spanning tree (NONE | EXISTING | ALL) (default : NONE)",
        )
        subparser.add_argument(
            "--keep_deprels",
            default="NONE",
            help="whether to keep current deprels and not predict new ones (NONE | "
            "EXISTING | ALL) (default : NONE)",
        )
        subparser.add_argument(
            "--keep_upos",
            default="NONE",
            help="whether to keep current upos and not predict new ones (NONE | "
            "EXISTING | ALL) (default : NONE)",
        )
        subparser.add_argument(
            "--keep_xpos",
            default="NONE",
            help="whether to keep current xpos and not predict new ones (NONE | "
            "EXISTING | ALL) (default : NONE)",
        )
        subparser.add_argument(
            "--keep_feats",
            default="NONE",
            help="whether to keep current feats and not predict new ones (NONE | "
            "EXISTING | ALL) (default : NONE)",
        )
        subparser.add_argument(
            "--keep_lemmas",
            default="NONE",
            help="whether to keep current lemmas and not predict new ones (NONE | "
            "EXISTING | ALL) (default : NONE)",
        )

        return subparser

    def run(self, args: Namespace):
        super().run(args)

        model_params = ModelParams_T.from_model_path(args.model_path)

        if args.batch_size:
            model_params.batch_size = args.batch_size

        in_to_out_paths, partial_pred_config = self.__validate_args(args)

        model = BertForDeprel.load_pretrained_for_prediction(
            args.model_path, args.device_config.device
        )

        predictor = Predictor(
            model,
            PredictionConfig(
                batch_size=model_params.batch_size, num_workers=args.num_workers
            ),
            args.device_config.multi_gpu,
        )

        print("Starting Predictions ...")
        for in_path, out_path in in_to_out_paths.items():
            print(f"Loading dataset from {in_path}...")

            sentences = load_conllu_sentences(in_path)
            pred_dataset = UDDataset(
                sentences,
                model.annotation_schema,
                model.embedding_type,
                model.max_position_embeddings,
                "train",
            )

            predicted_sentences, elapsed_seconds = predictor.predict(
                pred_dataset, partial_pred_config
            )

            writeConlluFile(out_path, predicted_sentences, overwrite=args.overwrite)

            print(
                f"Finished predicting `{out_path}, wrote {len(predicted_sentences)} "
                f"sents in {elapsed_seconds} secs`"
            )

            return predicted_sentences

    def __validate_args(self, args: Namespace):
        if not (args.model_path / "config.json").is_file():
            raise Exception(f"no config.json found in {args.model_path}")
        if not (args.model_path / "model.pt").is_file():
            raise Exception(f"no model.pt found in {args.model_path}")

        if args.num_workers < 0:
            raise Exception("num_workers must be greater than or equal to 0")

        output_dir: Path = args.outpath
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)

        unvalidated_input_paths: List[Path] = []
        if args.inpath.is_dir():
            unvalidated_input_paths = resolve_conllu_paths(args.inpath)
        elif args.inpath.is_file():
            unvalidated_input_paths.append(args.inpath)
        else:
            raise BaseException(
                f"args.inpath must be a folder or a file; was {args.inpath}"
            )

        in_to_out_paths = {}
        for input_path in unvalidated_input_paths:
            output_path = output_dir / input_path.name.replace(
                ".conllu", args.suffix + ".conllu"
            )
            if not args.overwrite:
                if output_path.is_file():
                    print(
                        f"file '{output_path}' already exists and overwrite!=False, "
                        "skipping ..."
                    )
                    continue
            in_to_out_paths[input_path] = output_path

        partial_pred_config = PartialPredictionConfig(
            keep_upos=args.keep_upos,
            keep_xpos=args.keep_xpos,
            keep_feats=args.keep_feats,
            keep_deprels=args.keep_deprels,
            keep_heads=args.keep_heads,
            keep_lemmas=args.keep_lemmas,
        )

        return in_to_out_paths, partial_pred_config


class Predictor:
    def __init__(
        self,
        model: BertForDeprel,
        config: PredictionConfig,
        multi_gpu: bool,
    ):
        self.config = config
        self.model = model
        self.multi_gpu = multi_gpu
        self.device = model.device

        if self.multi_gpu:
            print("MODEL TO MULTI GPU")
            self.model = nn.DataParallel(self.model)

        self.config = config
        self.data_loader_params = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
        }

    def predict(
        self,
        pred_dataset: UDDataset,
        partial_pred_config=PartialPredictionConfig(),
    ) -> Tuple[List[sentenceJson_T], float]:
        pred_loader = DataLoader(
            pred_dataset,
            collate_fn=pred_dataset.collate_fn_predict,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        print(
            f"Loaded {len(pred_dataset):5} sentences, "
            f"({len(pred_loader):3} batches)"
        )
        start = timer()
        predicted_sentences: List[sentenceJson_T] = []
        parsed_sentence_counter = 0
        batch: SequencePredictionBatch_T
        with torch.no_grad():
            for batch in pred_loader:
                preds = self.model.forward(batch).detach()

                time_from_start = 0
                parsing_speed = 0
                for predicted_sentence in self.__prediction_iterator(
                    batch, preds, pred_dataset, partial_pred_config
                ):
                    predicted_sentences.append(predicted_sentence)

                    parsed_sentence_counter += 1
                    time_from_start = timer() - start
                    parsing_speed = int(
                        round(
                            ((parsed_sentence_counter + 1) / time_from_start) / 100,
                            2,
                        )
                        * 100
                    )

                print(
                    "Predicting: "
                    f"{100 * (parsed_sentence_counter + 1)/len(pred_dataset):.2f}%"
                    f" complete. {time_from_start:.2f} seconds in file "
                    f"({parsing_speed} sents/sec)."
                )
        end = timer()
        elapsed_seconds = round(end - start, 2)
        return predicted_sentences, elapsed_seconds

    # TODO Next: explain this as much as possible
    # Then: combine this and the model eval chuliu_heads_pred method; it's the
    # same except for the constraint logic
    def __get_constrained_dependencies(
        self,
        heads_pred_sentence,
        deprels_pred,
        tok_starts_word,
        keep_heads: CopyOption,
        pred_dataset: UDDataset,
        n_sentence: int,
        idx_converter_sentence: Tensor,
    ):
        head_true_like = heads_pred_sentence.max(dim=0).indices
        chuliu_heads_pred = head_true_like.clone().cpu().numpy()
        chuliu_heads_list: List[int] = []

        # clone and set the value for the leading CLS token to True so that
        # Chu-Liu/Edmonds has the dummy root node it requires.
        tok_starts_word_or_is_root = tok_starts_word.clone()
        tok_starts_word_or_is_root[0] = True
        # Get the head scores for each word predicted
        heads_pred_np = heads_pred_sentence[:, tok_starts_word_or_is_root][
            tok_starts_word_or_is_root
        ]
        # Chu-Liu/Edmonds implementation requires numpy array, which can only be
        # created in CPU memory
        heads_pred_np = heads_pred_np.cpu().numpy()

        forced_relations: List[Tuple] = []
        if keep_heads == "EXISTING":
            forced_relations = pred_dataset.get_constrained_dependency_for_chuliu(
                n_sentence
            )

        # TODO: why transpose? C-L/E wants (dep, head), which is what we have. Unless
        # the constraint logic above is wrong, which is possible...
        chuliu_heads_vector = chuliu_edmonds_one_root_with_constraints(
            np.transpose(heads_pred_np, (1, 0)), forced_relations
        )
        # Remove the dummy root node for final output
        chuliu_heads_vector = chuliu_heads_vector[1:]
        for i_dependent_word, chuliu_head_pred in enumerate(chuliu_heads_vector):
            chuliu_heads_pred[
                idx_converter_sentence[i_dependent_word + 1]
            ] = idx_converter_sentence[chuliu_head_pred]
            chuliu_heads_list.append(int(chuliu_head_pred))

        # Move to same device for Tensor operations in _deprel_pred_for_heads
        chuliu_heads_pred = torch.tensor(chuliu_heads_pred).to(deprels_pred.device)
        # _depred_pred_for_heads expects a batch dimension, so add a dummy one to the
        # inputs and then remove it from the output
        deprels_pred_chuliu = _deprel_pred_for_heads(
            deprels_pred.unsqueeze(0), chuliu_heads_pred.unsqueeze(0)
        ).squeeze(0)

        # TODO: what are these return values?
        return chuliu_heads_list, deprels_pred_chuliu

    def __prediction_iterator(
        self,
        batch: SequencePredictionBatch_T,
        preds: BertForDeprelBatchOutput,
        pred_dataset: UDDataset,
        partial_pred_config: PartialPredictionConfig,
    ):
        idx_batch = batch.idx

        for i_sentence in range(batch.sequence_token_ids.size()[0]):
            raw_sentence_preds = preds.distributions_for_sentence(i_sentence)

            # TODO Next: encapsulate below in the output classes;
            # these will then be containers for the raw model outputs, with methods for
            # constructing the final predictions. The overwrite logic should be done
            # in a separate step, I think. Start by encapsulating the n_sentence and
            # pred_dataset stuff into the output classes.
            sentence_idx = idx_batch[i_sentence]
            n_sentence = int(sentence_idx)

            (
                chuliu_heads_list,
                deprels_pred_chuliu,
            ) = self.__get_constrained_dependencies(
                heads_pred_sentence=raw_sentence_preds.heads,
                deprels_pred=raw_sentence_preds.deprels,
                tok_starts_word=raw_sentence_preds.tok_starts_word,
                pred_dataset=pred_dataset,
                keep_heads=partial_pred_config.keep_heads,
                n_sentence=n_sentence,
                idx_converter_sentence=raw_sentence_preds.idx_converter,
            )

            # predictions for tokens that begin words are used as the predictions for
            # the words
            mask = raw_sentence_preds.tok_starts_word
            deprels_pred_chuliu_list = (
                deprels_pred_chuliu.max(dim=0).indices[mask].tolist()
            )

            uposs_pred_list = raw_sentence_preds.uposs.max(dim=1).indices[mask].tolist()

            xposs_pred_list = raw_sentence_preds.xposs.max(dim=1).indices[mask].tolist()

            feats_pred_list = raw_sentence_preds.feats.max(dim=1).indices[mask].tolist()

            lemma_scripts_pred_list = (
                raw_sentence_preds.lemma_scripts.max(dim=1).indices[mask].tolist()
            )

            yield pred_dataset.construct_sentence_prediction(
                n_sentence,
                uposs_pred_list,
                xposs_pred_list,
                chuliu_heads_list,
                deprels_pred_chuliu_list,
                feats_pred_list,
                lemma_scripts_pred_list,
                partial_pred_config=partial_pred_config,
            )
