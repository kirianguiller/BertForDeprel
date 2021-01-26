# -*- coding: utf-8 -*-

import os
import json
from collections import OrderedDict
from datetime import datetime
from parser.utils.os_utils import path_or_name

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split

from ..cmds.cmd import CMD
from ..utils.model_utils import BertForDeprel
from ..utils.load_data_utils import ConlluDataset, create_deprel_lists, create_pos_list
from ..utils.save import save_meta_model
from ..utils.train_utils import eval_epoch, train_epoch, update_history


class Train(CMD):
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(name, help="Train a model.")
        # subparser.add_argument('--buckets', default=32, type=int,
        #                        help='max num of buckets to use')
        subparser.add_argument(
            "--bert_type",
            "-b",
            default="bert-multilingual-cased",
            help="bert type to use (see huggingface models list)",
        )
        subparser.add_argument(
            "--batch_size", default=16, type=int, help="batch_size to use"
        )
        subparser.add_argument(
            "--punct", action="store_true", help="wether to include punctuation"
        )
        subparser.add_argument("--ftrain", required=True, help="path to train file")
        subparser.add_argument("--fdev", default="", help="path to dev file")
        subparser.add_argument("--ftest", default="", help="path to test file")
        subparser.add_argument(
            "--freeze_bert", action="store_true", help="path to test file"
        )
        subparser.add_argument("--fpretrain", default="", help="path to pretrain model")
        subparser.add_argument(
            "--reinit_bert", action="store_true", help="path to test file"
        )
        subparser.add_argument(
            "--compute_fields",
            action="store_true",
            help="compute new fields from conllu or not",
        )
        subparser.add_argument(
            "--keep_epoch",
            action="store_true",
            help="compute new fields from conllu or not",
        )
        subparser.add_argument(
            "--split_deprel",
            action="store_true",
            help="wether to split relations for predicting",
        )
        subparser.add_argument(
            "--n_to_train",
            default=0,
            type=int,
            help="number of sequences to use for parsing (used in the experience 1/10/100/1000...",
        )
        subparser.add_argument(
            "--n_to_test",
            default=0,
            type=int,
            help="number of sequences to use for parsing (used in the experience 1/10/100/1000...",
        )
        # subparser.add_argument('--increment_unicode', action='store_true',
        #                         help='whether to increment the unicode')

        # subparser.add_argument('--fembed', default='data/glove.6B.100d.txt',
        #                        help='path to pretrained embeddings')
        # subparser.add_argument('--unk', default='unk',
        #                        help='unk token in pretrained embeddings')

        return subparser

    def __call__(self, args):
        super(Train, self).__call__(args)
        # if path_or_name(args.ftrain) == "name":
        #     args.ftrain = os.path.join(args.folder, "train", args.ftrain)

        original_args = args

        if args.fpretrain:
            checkpoint = torch.load(args.fpretrain, map_location=torch.device("cpu"))
            loaded_args = checkpoint["args"]
            loaded_args.n_pretrain_epoch = checkpoint["n_epoch"]
            loaded_args.from_pretrain = True
            loaded_args.folder = args.folder
            loaded_args.fpretrain = args.fpretrain
            loaded_args.ftrain = args.ftrain
            loaded_args.ftest = args.ftest
            loaded_args.batch_size = args.batch_size
            loaded_args.num_workers = args.num_workers
            loaded_args.device = args.device
            loaded_args.multi_gpu = args.multi_gpu
            loaded_args.epochs = args.epochs
            loaded_args.model = args.model
            loaded_args.keep_epoch = args.keep_epoch
            loaded_args.punct = args.punct
            loaded_args.split_deprel = args.split_deprel
            loaded_args.patience = args.patience
            loaded_args.epochs = args.epochs

            args = loaded_args
            self.args = args
        else:
            # load annotation schema
            with open(args.path_annotation_schema, "r") as infile:
                annotation_schema_json = json.load(infile)

            print(annotation_schema_json)

            list_deprel_main = annotation_schema_json["splitted_deprel"]["main"]
            list_deprel_aux = annotation_schema_json["splitted_deprel"]["aux"]
            list_pos = annotation_schema_json["upos"]

            args.list_deprel_main = list_deprel_main
            args.n_labels_main = len(list_deprel_main)

            if args.split_deprel:
                args.list_deprel_aux = list_deprel_aux
                args.n_labels_aux = len(list_deprel_aux)

            args.list_pos = list_pos
            args.n_pos = len(list_pos)

        self.load_tokenizer(args.bert_type)
        train_dataset = ConlluDataset(args.ftrain, self.tokenizer, args)
        args.drm2i = train_dataset.drm2i
        args.i2drm = train_dataset.i2drm

        if args.split_deprel:
            args.dra2i = train_dataset.dra2i
            args.i2dra = train_dataset.i2dra

        args.pos2i = train_dataset.pos2i
        args.i2pos = train_dataset.i2pos

        # prepare test dataset
        if args.ftest:
            test_dataset = ConlluDataset(args.ftest, self.tokenizer, args)

        else:
            train_size = int(len(train_dataset) * 0.9)
            test_size = len(train_dataset) - train_size
            train_dataset, test_dataset = random_split(
                train_dataset, [train_size, test_size]
            )

        ### for experience with only part of the dataset
        if original_args.n_to_train:
            train_size = original_args.n_to_train
            # if original_args.n_to_test :
            #     test_size = original_args.n_to_test
            # else :
            test_size = int(len(train_dataset)) - train_size
            train_dataset, _ = random_split(train_dataset, [train_size, test_size])

        params_train = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }

        train_loader = DataLoader(train_dataset, **params_train)

        params_test = params_train
        params_test["batch_size"] = args.batch_size
        test_loader = DataLoader(test_dataset, **params_test)

        print(
            f"{'train:':6} {len(train_dataset):5} sentences, "
            f"{len(train_loader):3} batches, "
        )
        print(
            f"{'test:':6} {len(test_dataset):5} sentences, "
            f"{len(test_loader):3} batches, "
        )

        print("Create the model")
        model = BertForDeprel(args)

        n_epoch_start = 0
        if args.fpretrain:
            print("loading pretrain model")
            ### To reactivate if probleme in the loading of the model states
            loaded_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k.replace("module.", "")
                loaded_state_dict[name] = v

            model.load_state_dict(loaded_state_dict)

            if args.keep_epoch == True:
                n_epoch_start = args.n_pretrain_epoch


        model.to(args.device)

        if args.multi_gpu:
            print("MODEL TO MULTI GPU")
            model = nn.DataParallel(model)

        criterions = {}
        criterions["head"] = nn.CrossEntropyLoss(ignore_index=args.maxlen - 1)
        criterions["deprel"] = nn.CrossEntropyLoss(ignore_index=-1)
        criterions["pos"] = nn.CrossEntropyLoss(ignore_index=-1)

        args.criterions = criterions

        args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        total_timer_start = datetime.now()
        epochs_no_improve = 0
        LAS_best = 0
        n_epoch = 0

        results = eval_epoch(model, test_loader, args)

        history = []
        history = update_history(history, results, n_epoch_start, args)

        for n_epoch in range(n_epoch_start + 1, args.epochs + 1):
            print("\n-----   Epoch {}   -----".format(n_epoch))

            train_epoch(model, n_epoch, train_loader, args)

            results = eval_epoch(model, test_loader, args, n_epoch)

            history = update_history(history, results, n_epoch, args)

            if results["LAS_epoch"] > LAS_best:
                LAS_best = results["LAS_epoch"]
                save_meta_model(model, n_epoch, LAS_best, args)
                epochs_no_improve = 0
                best_epoch = n_epoch
                print("best epoch so far")

            else:
                epochs_no_improve += 1
                print("no improvement since {} epoch".format(epochs_no_improve))
                if epochs_no_improve >= args.patience:
                    print(
                        "Earlystopping ({} epochs without improvement)".format(
                            args.patience
                        )
                    )
                    break

        total_timer_end = datetime.now()
        total_time_elapsed = total_timer_end - total_timer_start

        print("Training ended. Total time elapsed = {}".format(total_time_elapsed))
