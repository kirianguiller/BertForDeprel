import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TypeVar

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..cmds.cmd import CMD, SubparsersType
from ..modules.BertForDepRel import BertForDeprel
from ..utils.annotation_schema import compute_annotation_schema
from ..utils.gpu_utils import DeviceConfig
from ..utils.load_data_utils import ConlluDataset, load_conllu_sentences
from ..utils.types import ConfigJSONEncoder, ModelParams_T

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
            "--model_folder_path", "-f", type=Path, help="path to models folder"
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
            "--batch_size_eval", type=int, help="number of epoch to do maximum"
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
            "--conf_pretrain",
            type=Path,
            help="path to pretrain model config",
        )
        subparser.add_argument(
            "--overwrite_pretrain_classifiers",
            action="store_true",
            help="erase pretrained classifier heads and recompute annotation schema",
        )

        return subparser

    def run(self, args, model_params: ModelParams_T):
        super().run(args, model_params)

        assert model_params.model_folder_path is not None

        if args.model_folder_path:
            model_params.model_folder_path = args.model_folder_path

        if not model_params.model_folder_path.is_dir():
            model_params.model_folder_path.mkdir(parents=True)

        if args.embedding_type:
            model_params.embedding_type = args.embedding_type

        if args.max_epoch:
            model_params.max_epoch = args.max_epoch

        if args.patience:
            model_params.patience = args.patience

        pretrain_model_params: Optional[ModelParams_T] = None
        if args.conf_pretrain:
            # We are finetuning an existing BertForDeprel model, where a pretrain model
            # config is provided. We need to be sure that:
            # - the annotation schema of new model is the same as finetuned
            # - the new model doesnt erase the old one (root_path_folder + model_name
            # are different)
            # - the new model has same architecture as old one
            with open(args.conf_pretrain, "r") as f:
                pretrain_model_params_ = ModelParams_T.from_dict(json.loads(f.read()))
                if not args.overwrite_pretrain_classifiers:
                    model_params.annotation_schema = (
                        pretrain_model_params_.annotation_schema
                    )
                # TODO: there should be an else clause where we update the annotation
                # schema with new labels from the fine-tuning data.
                pretrain_model_params = pretrain_model_params_
            if (
                pretrain_model_params_.model_folder_path
                == model_params.model_folder_path
            ):
                assert Exception(
                    "The pretrained model and the new model have same full path. It's "
                    "not allowed as it would result in erasing the pretrained model"
                )

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

        model_params.annotation_schema = compute_annotation_schema(
            iter(train_sentences)
        )

        train_dataset = ConlluDataset(
            iter(train_sentences),
            model_params.annotation_schema,
            model_params.embedding_type,
            model_params.max_position_embeddings,
            "train",
        )

        test_dataset = ConlluDataset(
            iter(test_sentences),
            model_params.annotation_schema,
            model_params.embedding_type,
            model_params.max_position_embeddings,
            "train",
        )

        path_scores_history = model_params.model_folder_path / "scores.history.json"
        path_scores_best = model_params.model_folder_path / "scores.best.json"

        total_timer_start = datetime.now()

        trainer = Trainer(
            model_params,
            args.device_config,
            pretrain_model_params,
            args.overwrite_pretrain_classifiers,
        )

        # set to infinity
        best_loss = float("inf")
        best_LAS = float("-inf")
        best_epoch_results = None
        epochs_without_improvement = 0
        history = []
        for epoch_results in trainer.train(
            train_dataset, test_dataset, args.batch_size_eval
        ):
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
                trainer.model.save_model(epoch_results["epoch"])  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
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

        path_finished_state_file = model_params.model_folder_path / ".finished"

        with open(path_finished_state_file, "w") as outfile:
            outfile.write("")


class Trainer:
    def __init__(
        self,
        model_params: ModelParams_T,
        # TODO: pass around single device config, not (device, multi_gpu)
        device_config: DeviceConfig = DeviceConfig(torch.device("cpu"), False),
        pretrain_model_params: Optional[ModelParams_T] = None,
        overwrite_pretrain_classifiers=True,
        num_workers=1,
    ):
        self.model_params = model_params
        self.device_config = device_config
        self.dataloader_params = {
            "batch_size": self.model_params.batch_size,
            "num_workers": num_workers,
            "shuffle": True,
        }

        print("Creating model for training...")
        self.model = BertForDeprel(
            model_params,
            pretrain_model_params=pretrain_model_params,
            overwrite_pretrain_classifiers=overwrite_pretrain_classifiers,
        )

        self.model = self.model.to(self.device_config.device)

        if self.device_config.multi_gpu:
            print("MODEL TO MULTI GPU")
            self.model = nn.DataParallel(self.model)

    def train(
        self,
        train_dataset: ConlluDataset,
        test_dataset: ConlluDataset,
        batch_size_eval=0,
    ):
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_fn_train,
            **self.dataloader_params,
        )
        params_test = self.dataloader_params.copy()
        if batch_size_eval:
            params_test["batch_size"] = batch_size_eval
        test_loader = DataLoader(
            test_dataset, collate_fn=train_dataset.collate_fn_train, **params_test
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

        no_train_results = self.model.eval_on_dataset(test_loader, self.device_config.device)  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
        no_train_results["n_sentences_train"] = len(train_dataset)
        no_train_results["n_sentences_test"] = len(test_dataset)
        no_train_results["epoch"] = n_epoch_start

        yield no_train_results

        for n_epoch in range(n_epoch_start + 1, self.model_params.max_epoch + 1):
            print(f"-----   Epoch {n_epoch}   -----")
            self.model.train_epoch(train_loader, self.device_config.device)  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
            results = self.model.eval_on_dataset(test_loader, self.device_config.device)  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
            results["n_sentences_train"] = len(train_dataset)
            results["n_sentences_test"] = len(test_dataset)
            results["epoch"] = n_epoch
            yield results
