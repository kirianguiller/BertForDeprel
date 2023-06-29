from argparse import _SubParsersAction, ArgumentParser
import os
from datetime import datetime
from time import time
import json
from typing import Optional

from torch import nn
from torch.utils.data import DataLoader, random_split

from ..cmds.cmd import CMD, SubparsersType
from ..utils.load_data_utils import ConlluDataset
from ..modules.BertForDepRel import BertForDeprel
from ..utils.types import ModelParams_T
from ..utils.scores_and_losses_utils import update_history
from ..utils.annotation_schema_utils import get_annotation_schema_from_input_folder, compute_annotation_schema, is_annotation_schema_empty

class Train(CMD):
    def add_subparser(self, name: str, parser: SubparsersType) -> ArgumentParser:
        subparser = parser.add_parser(name, help="Train a model.")
        subparser.add_argument(
            "--model_folder_path", "-f", help="path to models folder"
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
            "--ftrain", required=True, help="path to train file or folder (files need to have .conllu extension)")
        subparser.add_argument(
            "--ftest", default="", help="path to test file or folder (files need to have .conllu extension)")
        subparser.add_argument(
            "--split_ratio",
            default=0.8,
            type=float,
            help="split ratio to use (if no --ftest is provided)",
        )
        subparser.add_argument(
            "--path_annotation_schema", help="path to annotation schema (default : in folder/annotation_schema.json"
        )

        subparser.add_argument(
            "--path_folder_compute_annotation_schema", help="path to annotation schema (default : in folder/annotation_schema.json"
        )
        subparser.add_argument(
            "--conf_pretrain", default="", help="path to pretrain model config")
        subparser.add_argument(
            "--overwrite_pretrain_classifiers", action="store_true", help="erase pretraines classifier heads and recompute annotation schema")

        return subparser

    def __call__(self, args, model_params: ModelParams_T):
        super(Train, self).__call__(args, model_params)
        if args.model_folder_path:
            model_params["model_folder_path"] = args.model_folder_path

        if not os.path.isdir(model_params["model_folder_path"]):
            os.makedirs(model_params["model_folder_path"])

        if args.embedding_type:
            model_params["embedding_type"] = args.embedding_type

        if args.max_epoch:
            model_params["max_epoch"] = args.max_epoch

        if args.patience:
            model_params["patience"] = args.patience

        # if user provided a path to an annotation schema, use this one (or overwrite current one if it exits)
        if args.path_annotation_schema:
            if args.path_folder_compute_annotation_schema:
                raise Exception("You provided both --path_annotation_schema and --path_folder_compute_annotation_schema, it's not allowed as it is ambiguous. You can provide none of them or at maximum one of this two.")
            print(f"You provided a path to a custom annotation schema, we will use this one for your model `{args.path_annotation_schema}`")
            with open(args.path_annotation_schema, "r") as infile:
                model_params["annotation_schema"] = json.loads(infile.read())

        # if no annotation schema where provided, either
        if args.path_folder_compute_annotation_schema:
            model_params["annotation_schema"] = get_annotation_schema_from_input_folder(args.path_folder_compute_annotation_schema)
        print("Model parameters : ", model_params)

        # if is_annotation_schema_empty(model_params["annotation_schema"]) == True:
        #     # The annotation schema was never given in json config or path argument, we need to compute it on --ftrain
        #     print("Computing annotation schema on --ftrain file")
        #     model_params["annotation_schema"] = compute_annotation_schema(args.ftrain)

        pretrain_model_params: Optional[ModelParams_T] = None
        if args.conf_pretrain:
            # We are finetuning an existing BertForDeprel model, where a pretrain model config is provided
            # we need to be sure that :
            # - the annotation schema of new model is the same as finetuned
            # - the new model doesnt erase the old one (root_path_folder + model_name are different)
            # - the new model has same architecture as old one
            with open(args.conf_pretrain, "r") as infile:
                pretrain_model_params_: ModelParams_T = json.loads(infile.read())
                if args.overwrite_pretrain_classifiers == False:
                    model_params["annotation_schema"] = pretrain_model_params_["annotation_schema"]
                pretrain_model_params = pretrain_model_params_
            if os.path.join(pretrain_model_params_["model_folder_path"], "model.pt")  == \
                os.path.join(model_params["model_folder_path"], "model.pt"):
                assert Exception("The pretrained model and the new model have same full path. It's not allowed as it would result in erasing the pretrained model")

        dataset = ConlluDataset(args.ftrain, model_params, args.mode, compute_annotation_schema_if_not_found=True)

        # prepare test dataset
        if args.ftest:
            train_dataset = dataset
            test_dataset = ConlluDataset(args.ftest, model_params, args.mode)

        else:
            train_size = int(len(dataset) * args.split_ratio)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        params_train = {
            "batch_size": model_params["batch_size"],
            "num_workers": args.num_workers,
            "shuffle": True,
        }

        train_loader = DataLoader(
            train_dataset, collate_fn=dataset.collate_fn, **params_train
        )

        params_test = {
            "batch_size": model_params["batch_size"],
            "num_workers": args.num_workers,
            "shuffle": True,
        }
        if args.batch_size_eval:
            params_test["batch_size"] = args.batch_size_eval
        test_loader = DataLoader(
            test_dataset, collate_fn=dataset.collate_fn, **params_test
        )

        print(
            f"{'train:':6} {len(train_dataset):5} sentences, "
            f"{len(train_loader):3} batches, "
        )
        print(
            f"{'test:':6} {len(test_dataset):5} sentences, "
            f"{len(test_loader):3} batches, "
        )

        print("Create the model")
        model = BertForDeprel(model_params, pretrain_model_params=pretrain_model_params, overwrite_pretrain_classifiers=args.overwrite_pretrain_classifiers)

        n_epoch_start = 0
        model.to(args.device)

        if args.multi_gpu:
            print("MODEL TO MULTI GPU")
            model = nn.DataParallel(model)

        # scheduler = get_linear_schedule_with_warmup(args.optimizer, args.batch_per_epoch * 5, args.batch_per_epoch*args.epochs)

        total_timer_start = datetime.now()
        epochs_no_improve = 0
        n_epoch = 0

        results = model.eval_epoch(test_loader, args.device)
        results["n_sentences_train"] = len(train_dataset)
        results["n_sentences_test"] = len(test_dataset)
        results["epoch"] = n_epoch_start

        history = []
        history.append(results)
        best_loss = results["loss_epoch"]
        best_LAS = results["LAS_epoch"]
        best_epoch_results = results
        path_scores_history = os.path.join(model_params["model_folder_path"], "scores.history.json")
        path_scores_best = os.path.join(model_params["model_folder_path"], "scores.best.json")
        for n_epoch in range(n_epoch_start + 1, model_params["max_epoch"] + 1):
            print("\n-----   Epoch {}   -----".format(n_epoch))
            model.train_epoch(train_loader, args.device)
            results = model.eval_epoch(test_loader, args.device)
            results["n_sentences_train"] = len(train_dataset)
            results["n_sentences_test"] = len(test_dataset)
            results["epoch"] = n_epoch

            history.append(results)
            with open(path_scores_history, "w") as outfile:
                outfile.write(json.dumps(history, indent=4))

            # history = update_history(history, results, n_epoch, args)
            loss_epoch = results["loss_epoch"]
            LAS_epoch = results["LAS_epoch"]
            if loss_epoch < best_loss or LAS_epoch > best_LAS:
                epochs_no_improve = 0
                if loss_epoch < best_loss:
                    best_loss = loss_epoch
                    print("best epoch (on cumul loss) so far, saving model...")

                if LAS_epoch > best_LAS:
                    best_LAS = LAS_epoch
                    print("best epoch (on LAS) so far, saving model...")

                model.save_model(n_epoch)
                with open(path_scores_best, "w") as outfile:
                    outfile.write(json.dumps(results, indent=4))
                best_epoch_results = results

            else:
                epochs_no_improve += 1
                print("no improvement since {} epoch".format(epochs_no_improve))
                if epochs_no_improve >= model_params["patience"]:
                    print(
                        "Earlystopping ({} epochs without improvement)".format(
                            model_params["patience"]
                        )
                    )
                    print("\nbest result : ", best_epoch_results)
                    break
        total_timer_end = datetime.now()
        total_time_elapsed = total_timer_end - total_timer_start

        print("Training ended. Total time elapsed = {}".format(total_time_elapsed))

        path_finished_state_file = os.path.join(model_params["model_folder_path"], ".finished")
        with open(path_finished_state_file, "w") as outfile:
            outfile.write("")
