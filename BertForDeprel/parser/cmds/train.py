import json
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Generator, List, TypeVar

from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..cmds.cmd import CMD, SubparsersType
from ..modules.BertForDepRel import BertForDeprel, EvalResult, TrainingDiagnostics
from ..utils.annotation_schema import compute_annotation_schema
from ..utils.load_data_utils import load_conllu_sentences
from ..utils.types import DataclassJSONEncoder, ModelParams_T, TrainingConfig
from ..utils.ud_dataset import UDDataset

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

        try:
            model_params = ModelParams_T.from_model_path(args.new_model_path)
        except FileNotFoundError:
            model_params = ModelParams_T()

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

        if args.num_workers:
            model_params.num_workers = args.num_workers

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

        if args.pretrained_path:
            print("Loading pretrained model...")
            if args.pretrained_path.resolve() == args.new_model_path.resolve():
                raise ValueError(
                    "The pretrained model and the new model have same full path. It's "
                    "not allowed as it would result in erasing the pretrained model."
                )
            if args.overwrite_pretrain_classifiers:
                model = BertForDeprel.load_pretrained_for_retraining(
                    args.pretrained_path,
                    train_data_annotation_schema,
                    args.device_config.device,
                )
            else:
                model = BertForDeprel.load_pretrained_for_finetuning(
                    args.pretrained_path,
                    train_data_annotation_schema,
                    args.device_config.device,
                )
        else:
            print("Creating model for training...")
            model = BertForDeprel.new_model(
                model_params.embedding_type,
                train_data_annotation_schema,
                args.device_config.device,
            )

        model.add_diagnostic("training_command", sys.argv)

        train_dataset = model.encode_dataset(iter(train_sentences))
        test_dataset = model.encode_dataset(iter(test_sentences))

        training_config = TrainingConfig(
            max_epochs=model_params.max_epoch,
            patience=model_params.patience,
            batch_size=model_params.batch_size,
            num_workers=model_params.num_workers,
        )
        trainer = Trainer(
            training_config,
            args.device_config.multi_gpu,
        )

        trainer.train_until_quiescence(
            model, train_dataset, test_dataset, args.new_model_path
        )


# TODO: not really clear that this needs to be an object; we can pass the model and
# config as arguments to the train method instead and it wouldn't make much difference.
class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        multi_gpu: bool = False,
    ):
        self.multi_gpu = multi_gpu
        if self.multi_gpu:
            print("MODEL TO MULTI GPU")
            self.model = nn.DataParallel(self.model)

        self.config = config

    def train_until_quiescence(
        self,
        model: BertForDeprel,
        train_dataset: UDDataset,
        test_dataset: UDDataset,
        output_dir: Path,
    ) -> EvalResult:
        """Train the model until it stops improving or we've reached the specified
        maximum number of epochs, then return the best result. Write the best model
        to file, as well as the history of scores and the scores of the best model.
        When finished, write a .finished file to the output directory. This can be used
        to check if the training was interrupted, and also for signaling to other
        processes that the training is finished."""
        path_scores_history = output_dir / "scores.history.json"
        path_scores_best = output_dir / "scores.best.json"

        best_loss = float("inf")
        best_LAS = float("-inf")
        best_epoch_results = None
        epochs_without_improvement = 0

        history = []
        total_timer_start = datetime.now()
        for epoch, epoch_results in enumerate(
            self.train(model, train_dataset, test_dataset)
        ):
            print(epoch_results)

            loss_epoch = epoch_results.loss_epoch
            LAS_epoch = epoch_results.LAS_epoch

            is_best_loss = loss_epoch < best_loss
            is_best_LAS = LAS_epoch > best_LAS
            saved = False
            stopping_early = False

            if is_best_loss or is_best_LAS:
                epochs_without_improvement = 0
                if is_best_loss:
                    best_loss = loss_epoch
                    print("best epoch loss so far")

                if is_best_LAS:
                    best_LAS = LAS_epoch
                    print("best epoch LAS so far")

                print("Saving model")
                model.save(  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
                    output_dir, self.config
                )
                saved = True
                best_epoch_results = epoch_results
            else:
                epochs_without_improvement += 1
                print(
                    "no improvement since {} epoch".format(epochs_without_improvement)
                )
                if epochs_without_improvement >= self.config.patience:
                    print(
                        "Stopping early ({} epochs without improvement)".format(
                            self.config.patience
                        )
                    )
                    print("\nbest result : ", best_epoch_results)
                    stopping_early = True

            diagnostics = TrainingDiagnostics(
                n_sentences_train=len(train_dataset),
                n_sentences_test=len(test_dataset),
                epoch=epoch,
                saved=saved,
                is_best_loss=is_best_loss,
                is_best_LAS=is_best_LAS,
                epochs_without_improvement=epochs_without_improvement,
                stopping_early=stopping_early,
            )
            epoch_results._set_diagnostic_info(diagnostics)
            history.append(epoch_results.rounded(3))
            with open(path_scores_history, "w") as outfile:
                outfile.write(json.dumps(history, indent=4, cls=DataclassJSONEncoder))

            if saved:
                with open(path_scores_best, "w") as outfile:
                    outfile.write(
                        json.dumps(epoch_results, indent=4, cls=DataclassJSONEncoder)
                    )

            if stopping_early or epoch >= self.config.max_epochs:
                break

        total_timer_end = datetime.now()
        total_time_elapsed = total_timer_end - total_timer_start

        print("Training finished. Total time elapsed = {}".format(total_time_elapsed))
        with (output_dir / ".finished").open("w") as outfile:
            outfile.write("")

        assert best_epoch_results is not None
        return best_epoch_results

    def train(
        self,
        model: BertForDeprel,
        train_dataset: UDDataset,
        test_dataset: UDDataset,
    ) -> Generator[EvalResult, None, None]:
        """Train the model on the given dataset. Yields the results of each epoch, with
        no stopping criteria (client must decide when to stop consuming the generator).
        """
        train_loader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.collate_train,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset,
            collate_fn=train_dataset.collate_train,
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

        def announce_epoch(epoch: int) -> None:
            print(f"-----   Epoch {epoch}   -----")

        epoch = 0
        announce_epoch(epoch)
        no_train_results: EvalResult = model.eval_on_dataset(  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
            test_loader
        )
        yield no_train_results

        while True:
            epoch += 1
            announce_epoch(epoch)
            model.train_epoch(  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
                train_loader
            )
            results = model.eval_on_dataset(  # type: ignore (https://github.com/pytorch/pytorch/issues/90827) # noqa: E501
                test_loader
            )
            yield results
