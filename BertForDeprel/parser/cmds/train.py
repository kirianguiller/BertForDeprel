import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, TypeVar

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..cmds.cmd import CMD, SubparsersType
from ..modules.BertForDepRel import BertForDeprel
from ..utils.annotation_schema import compute_annotation_schema
from ..utils.gpu_utils import DeviceConfig
from ..utils.load_data_utils import UDDataset, load_conllu_sentences
from ..utils.types import ConfigJSONEncoder, ModelParams_T, TrainingConfig

T = TypeVar("T")


class ListDataset(Dataset[T]):
    """Dummy wrapper for when we need a Dataset but have a list"""

    def __init__(self, sentences: List[T]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx) -> T:
        return self.sentences[idx]


class TrainCmd(CMD):
    def add_subparser(self, name: str, parser: SubparsersType) -> ArgumentParser:
        subparser = parser.add_parser(name, help="Train a model.")
        subparser.add_argument(
            "--new_model_path", "-f", type=Path, help="Path to write new model to"
        )
        subparser.add_argument(
            "--embedding_type",
            "-e",
            help="bert type to use (see huggingface models list)",
        )
        subparser.add_argument(
            "--max_epoch", type=int, help="number of epoch to do maximum"
        )
        subparser.add_argument(
            "--patience", type=int, help="number of epoch to do maximum"
        )
        subparser.add_argument(
            "--ftrain",
            required=True,
            type=Path,
            help="path to train file or folder (files need to have .conllu extension)",
        )
        subparser.add_argument(
            "--ftest",
            type=Path,
            help="path to test file or folder (files need to have .conllu extension)",
        )
        subparser.add_argument(
            "--split_ratio",
            default=0.8,
            type=float,
            help="split ratio to use (if no --ftest is provided)",
        )

        subparser.add_argument(
            "--pretrained_path",
            type=Path,
            help="Path of pretrained model",
        )
        subparser.add_argument(
            "--overwrite_pretrain_classifiers",
            action="store_true",
            help="erase pretrained classifier heads and recompute annotation schema",
        )

        return subparser

    def run(self, args):
        super().run(args)

        with open(args.new_model_path / "config.json", "r") as f:
            model_params = ModelParams_T.from_dict(json.load(f))

        if not args.new_model_path.is_dir():
            args.new_model_path.mkdir(parents=True)

        if args.embedding_type:
            model_params.embedding_type = args.embedding_type

        if args.max_epoch:
            model_params.max_epoch = args.max_epoch

        if args.patience:
            model_params.patience = args.patience

        if args.batch_size:
            model_params.batch_size = args.batch_size

        train_sentences = load_conllu_sentences(args.ftrain)

        if args.ftest:
            print(f"Using {args.ftrain} for training and {args.ftest} for testing")
            test_sentences = load_conllu_sentences(args.ftest)
        else:
            print(
                f"Splitting {args.ftrain} into train and test sets with ratio "
                f"{args.split_ratio}"
            )
            train_size = int(len(train_sentences) * args.split_ratio)
            test_size = len(train_sentences) - train_size
            train_sentences, test_sentences = random_split(
                dataset=ListDataset(train_sentences), lengths=[train_size, test_size]
            )  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501

        train_data_annotation_schema = compute_annotation_schema(iter(train_sentences))

        path_scores_history = args.new_model_path / "scores.history.json"
        path_scores_best = args.new_model_path / "scores.best.json"

        if args.pretrained_path:
            print("Loading pretrained model...")
            if args.pretrained_path.resolve() == args.new_model_path.resolve():
                raise ValueError(
                    "The pretrained model and the new model have same full path. It's "
                    "not allowed as it would result in erasing the pretrained model."
                )
            if args.overwrite_pretrain_classifiers:
                model = BertForDeprel.load_pretrained_for_retraining(
                    args.pretrained_path, train_data_annotation_schema
                )
            else:
                model = BertForDeprel.load_pretrained_for_finetuning(
                    args.pretrained_path,
                    train_data_annotation_schema,
                )
        else:
            print("Creating model for training...")
            model = BertForDeprel.new_model(
                model_params.embedding_type, train_data_annotation_schema
            )

        train_dataset = UDDataset(
            iter(train_sentences),
            model.annotation_schema,
            model.embedding_type,
            model.max_position_embeddings,
            "train",
        )

        test_dataset = UDDataset(
            iter(test_sentences),
            model.annotation_schema,
            model.embedding_type,
            model.max_position_embeddings,
            "train",
        )

        training_config = TrainingConfig(
            max_epochs=model_params.max_epoch,
            patience=model_params.patience,
            batch_size=model_params.batch_size,
            num_workers=args.num_workers,
        )
        trainer = Trainer(
            model,
            training_config,
            args.device_config,
        )

        # TODO: move this logic into the Trainer class

        # set to infinity
        best_loss = float("inf")
        best_LAS = float("-inf")
        best_epoch_results = None
        epochs_without_improvement = 0
        history = []
        total_timer_start = datetime.now()
        for epoch_results in trainer.train(train_dataset, test_dataset):
            history.append(epoch_results)
            with open(path_scores_history, "w") as outfile:
                outfile.write(json.dumps(history, indent=4, cls=ConfigJSONEncoder))

            loss_epoch = epoch_results["loss_epoch"]
            LAS_epoch = epoch_results["LAS_epoch"]
            if loss_epoch < best_loss or LAS_epoch > best_LAS:
                epochs_without_improvement = 0
                if loss_epoch < best_loss:
                    best_loss = loss_epoch
                    print("best epoch (on cumul loss) so far")

                if LAS_epoch > best_LAS:
                    best_LAS = LAS_epoch
                    print("best epoch (on LAS) so far")

                print("Saving model")
                trainer.model.save_model(  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
                    args.new_model_path, training_config, int(epoch_results["epoch"])
                )
                with open(path_scores_best, "w") as outfile:
                    outfile.write(
                        json.dumps(epoch_results, indent=4, cls=ConfigJSONEncoder)
                    )
                best_epoch_results = epoch_results
            else:
                epochs_without_improvement += 1
                print(
                    "no improvement since {} epoch".format(epochs_without_improvement)
                )
                if epochs_without_improvement >= model_params.patience:
                    print(
                        "Earlystopping ({} epochs without improvement)".format(
                            model_params.patience
                        )
                    )
                    print("\nbest result : ", best_epoch_results)
                    break

        total_timer_end = datetime.now()
        total_time_elapsed = total_timer_end - total_timer_start

        print("Training ended. Total time elapsed = {}".format(total_time_elapsed))

        path_finished_state_file = args.new_model_path / ".finished"

        with open(path_finished_state_file, "w") as outfile:
            outfile.write("")


# TODO: not really clear that this needs to be an object; we can pass the model and
# config as arguments to the train method instead and it wouldn't make much difference.
class Trainer:
    def __init__(
        self,
        model: BertForDeprel,
        config: TrainingConfig,
        device_config: DeviceConfig = DeviceConfig(torch.device("cpu"), False),
    ):
        self.device_config = device_config
        self.model = model.to(self.device_config.device)
        if self.device_config.multi_gpu:
            print("MODEL TO MULTI GPU")
            self.model = nn.DataParallel(self.model)

        self.config = config

    def train(
        self,
        train_dataset: UDDataset,
        test_dataset: UDDataset,
    ) -> Generator[Dict[str, float], None, None]:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn_train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset,
            collate_fn=train_dataset.collate_fn_train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

        print(
            f"{'train:':6} {len(train_dataset):5} sentences, "
            f"{len(train_loader):3} batches, "
        )
        print(
            f"{'test:':6} {len(test_dataset):5} sentences, "
            f"{len(test_loader):3} batches, "
        )

        n_epoch_start = 0

        no_train_results: Dict[str, float] = self.model.eval_on_dataset(test_loader, self.device_config.device)  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
        no_train_results["n_sentences_train"] = len(train_dataset)
        no_train_results["n_sentences_test"] = len(test_dataset)
        no_train_results["epoch"] = n_epoch_start

        yield no_train_results

        for n_epoch in range(n_epoch_start + 1, self.config.max_epochs + 1):
            print(f"-----   Epoch {n_epoch}   -----")
            self.model.train_epoch(train_loader, self.device_config.device)  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
            results = self.model.eval_on_dataset(test_loader, self.device_config.device)  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
            results["n_sentences_train"] = len(train_dataset)
            results["n_sentences_test"] = len(test_dataset)
            results["epoch"] = n_epoch
            yield results
