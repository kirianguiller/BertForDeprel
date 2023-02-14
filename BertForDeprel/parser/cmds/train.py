# -*- coding: utf-8 -*-
from datetime import datetime
from time import time

from torch import nn
from torch.utils.data import DataLoader, random_split

from transformers import AutoTokenizer

from ..cmds.cmd import CMD
from ..utils.load_data_utils import ConlluDataset
from ..utils.model_utils import BertForDeprel
from ..utils.types import ModelParams_T
from ..utils.scores_and_losses_utils import update_history


class Train(CMD):
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(name, help="Train a model.")
        # subparser.add_argument('--buckets', default=32, type=int,
        #                        help='max num of buckets to use')
        subparser.add_argument(
            "--bert_type",
            "-b",
            help="bert type to use (see huggingface models list)",
        )
        subparser.add_argument(
            "--batch_size", type=int, help="batch_size to use"
        )
        subparser.add_argument(
            "--max_epoch", type=int, help="number of epoch to do maximum"
        )
        subparser.add_argument("--ftrain", required=True, help="path to train file")
        subparser.add_argument("--fdev", default="", help="path to dev file")
        subparser.add_argument("--ftest", default="", help="path to test file")
        subparser.add_argument(
            "--split_ratio",
            default=0.8,
            type=float,
            help="split ratio to use (if no --ftest is provided)",
        )
        # subparser.add_argument("--fpretrain", default="", help="path to pretrain model")
        subparser.add_argument(
            "--keep_epoch",
            action="store_true",
            help="keep previous numpr of epochs if pretrained",
        )
        return subparser

    def __call__(self, args, model_params: ModelParams_T):
        super(Train, self).__call__(args, model_params)

        if args.max_epoch:
            model_params["max_epoch"] = args.max_epoch

        if args.batch_size:
            model_params["batch_size"] = args.batch_size

        annotation_schema_json = model_params.get("annotation_schema")
        if annotation_schema_json is None:
            # TODO : Implement the auto-annotation schema
            raise Exception(
                "You didnt provide an annotation schema. The auto-annotation schema computing is still not implemented"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_params["embedding_type"])
        dataset = ConlluDataset(args.ftrain, tokenizer, model_params, args.mode)
        model_params["annotation_schema"]["dep2i"] = dataset.dep2i
        model_params["annotation_schema"]["i2dep"] = dataset.i2dep

        model_params["annotation_schema"]["pos2i"] = dataset.pos2i
        model_params["annotation_schema"]["i2pos"] = dataset.i2pos

        # prepare test dataset
        if args.ftest:
            train_dataset = dataset
            test_dataset = ConlluDataset(args.ftest, tokenizer, model_params, args.mode)

        else:
            train_size = int(len(dataset) * args.split_ratio)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # ### for experience with only part of the dataset
        # if original_args.n_to_train:
        #     train_size = original_args.n_to_train
        #     test_size = int(len(train_dataset)) - train_size
        #     train_dataset, _ = random_split(train_dataset, [train_size, test_size])

        params_train = {
            "batch_size": model_params["batch_size"],
            "num_workers": args.num_workers,
        }

        train_loader = DataLoader(
            train_dataset, collate_fn=dataset.collate_fn, **params_train
        )

        params_test = params_train
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
        model = BertForDeprel(model_params)

        n_epoch_start = 0
        # if args.fpretrain:
        #     print("loading pretrain model")
        #     ### To reactivate if probleme in the loading of the model states
        #     loaded_state_dict = OrderedDict()
        #     for k, v in checkpoint["state_dict"].items():
        #         name = k.replace("module.", "")
        #         loaded_state_dict[name] = v

        #     model.load_state_dict(loaded_state_dict)

        #     if args.keep_epoch == True:
        #         n_epoch_start = args.n_pretrain_epoch

        model.to(args.device)

        if args.multi_gpu:
            print("MODEL TO MULTI GPU")
            model = nn.DataParallel(model)

        # args.batch_per_epoch = len(train_loader)
        # scheduler = get_linear_schedule_with_warmup(args.optimizer, args.batch_per_epoch * 5, args.batch_per_epoch*args.epochs)

        total_timer_start = datetime.now()
        epochs_no_improve = 0
        n_epoch = 0

        results = model.eval_epoch(test_loader, args.device)
        best_loss = results["loss_epoch"]
        best_LAS = results["LAS_epoch"]
        best_epoch_results = results

        # history = []
        # history = update_history(history, results, n_epoch_start, args)
        t_total_train = 0.
        t_total_eval = 0.
        t_total_evalsaving = 0.
        for n_epoch in range(n_epoch_start + 1, model_params["max_epoch"] + 1):
            print("\n-----   Epoch {}   -----".format(n_epoch))
            t_before_train = time()
            model.train_epoch(train_loader, args.device)
            t_after_train = time()
            t_total_train += t_after_train - t_before_train

            t_before_eval = time()
            t_before_evalsaving = time()

            results = model.eval_epoch(test_loader, args.device)
            t_after_eval = time()
            t_total_eval += t_after_eval - t_before_eval
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
            t_after_evalsaving = time()
            t_total_evalsaving += t_after_evalsaving - t_before_evalsaving
            print(
                "KK timers : ",
                round(t_total_train, 4),
                round(t_total_eval, 4),
                round(t_total_evalsaving, 4),
            )

        total_timer_end = datetime.now()
        total_time_elapsed = total_timer_end - total_timer_start

        print("Training ended. Total time elapsed = {}".format(total_time_elapsed))
