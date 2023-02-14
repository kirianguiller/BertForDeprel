# -*- coding: utf-8 -*-

import os
import json
from collections import OrderedDict
from datetime import datetime
from parser.utils.os_utils import path_or_name
from time import time

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW

from ..cmds.cmd import CMD
from ..utils.load_data_utils import ConlluDataset
from ..utils.model_utils import BertForDeprel
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
            "--epochs", default=10, type=int, help="batch_size to use"
        )
        subparser.add_argument(
            "--punct", action="store_true", help="wether to include punctuation"
        )
        subparser.add_argument("--ftrain", required=True, help="path to train file")
        subparser.add_argument("--fdev", default="", help="path to dev file")
        subparser.add_argument("--ftest", default="", help="path to test file")
        subparser.add_argument(
            "--split_ratio", default=0.8, type=float, help="split ratio to use (if no --ftest is provided)"
        )
        subparser.add_argument(
            "--freeze_bert", action="store_true", help="path to test file"
        )
        subparser.add_argument("--fpretrain", default="", help="path to pretrain model")
        subparser.add_argument(
            "--reinit_bert", action="store_true", help="path to test file"
        )
        subparser.add_argument(
            "--keep_epoch",
            action="store_true",
            help="keep previous numpr of epochs if pretrained",
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
        return subparser

    def __call__(self, args):
        super(Train, self).__call__(args)

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

            list_deprel_full = annotation_schema_json["deprel"]
            list_deprel_main = annotation_schema_json["splitted_deprel"]["main"]
            list_deprel_aux = annotation_schema_json["splitted_deprel"]["aux"]
            list_pos = annotation_schema_json["upos"]
            list_lemma_script = annotation_schema_json["lemma_script"]
            print("\nKK len list_lemma_script", len(list_lemma_script))
            if not args.split_deprel:
                list_deprel_main = list_deprel_full
                list_deprel_aux = list_deprel_full

            else:
                list_deprel_main = list_deprel_main
                list_deprel_aux = list_deprel_aux


            args.list_deprel_main = list_deprel_main
            args.n_labels_main = len(list_deprel_main)

            args.list_deprel_aux = list_deprel_aux
            args.n_labels_aux = len(list_deprel_aux)

            args.list_pos = list_pos
            args.n_pos = len(list_pos)

            args.list_lemma_script = list_lemma_script
            args.n_lemma_script = len(list_lemma_script)

        args.batch_size = 16
        self.load_tokenizer(args.bert_type)
        dataset = ConlluDataset(args.ftrain, self.tokenizer, args)
        args.drm2i = dataset.drm2i
        args.i2drm = dataset.i2drm

        if args.split_deprel:
            args.dra2i = dataset.dra2i
            args.i2dra = dataset.i2dra

        args.pos2i = dataset.pos2i
        args.i2pos = dataset.i2pos

        args.lemma_script2i = dataset.lemma_script2i
        args.i2lemma_script = dataset.i2lemma_script

        # prepare test dataset
        if args.ftest:
            train_dataset = dataset
            test_dataset = ConlluDataset(args.ftest, self.tokenizer, args)

        else:
            train_size = int(len(dataset) * args.split_ratio)
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(
                dataset, [train_size, test_size]
            )

        ### for experience with only part of the dataset
        if original_args.n_to_train:
            train_size = original_args.n_to_train
            test_size = int(len(train_dataset)) - train_size
            train_dataset, _ = random_split(train_dataset, [train_size, test_size])

        params_train = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }

        train_loader = DataLoader(train_dataset, collate_fn=dataset.collate_fn, **params_train)

        params_test = params_train
        params_test["batch_size"] = args.batch_size
        test_loader = DataLoader(test_dataset, collate_fn=dataset.collate_fn, **params_test)

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
        criterions["head"] = nn.CrossEntropyLoss(ignore_index=-1)
        criterions["deprel"] = nn.CrossEntropyLoss(ignore_index=-1)
        criterions["pos"] = nn.CrossEntropyLoss(ignore_index=-1)
        criterions["lemma_script"] = nn.CrossEntropyLoss(ignore_index=-1)

        args.criterions = criterions
        # print("KK model.parameters()", model.parameters())

        # model_parameters = [(n, p) for (n, p) in model.llm_layer.named_parameters()] + \
        #                    [(n, p) for (n, p) in model.tagger_layer.named_parameters()]
        # param_groups = [
        #         {
        #             'params': [p for n, p in model_parameters if 'adapters' in n if
        #                        p.requires_grad],
        #             'lr': 0.0001, 'weight_decay': 0.0001
        #         },
        #         {
        #             'params': [p for n, p in model_parameters if 'adapters' not in n if
        #                        p.requires_grad],
        #             'lr': 0.001, 'weight_decay': 0.001
        #         }
        #     ]

        # args.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        args.batch_per_epoch = len(train_loader)
        # scheduler = get_linear_schedule_with_warmup(args.optimizer, args.batch_per_epoch * 5, args.batch_per_epoch*args.epochs)

        total_timer_start = datetime.now()
        epochs_no_improve = 0
        n_epoch = 0

        results = eval_epoch(model, test_loader, args)
        best_loss = results["loss_epoch"]
        best_LAS = results["LAS_epoch"]
        best_epoch_results = results

        history = []
        history = update_history(history, results, n_epoch_start, args)
        t_total_train = 0
        t_total_eval = 0
        t_total_evalsaving = 0
        for n_epoch in range(n_epoch_start + 1, args.epochs + 1):
            print("\n-----   Epoch {}   -----".format(n_epoch))
            t_before_train = time()
            model.train_epoch(train_loader, args)
            t_after_train = time()
            t_total_train += t_after_train - t_before_train

            t_before_eval = time()
            t_before_evalsaving = time()

            results = eval_epoch(model, test_loader, args, n_epoch)
            t_after_eval = time()
            t_total_eval += t_after_eval - t_before_eval
            history = update_history(history, results, n_epoch, args)
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

                save_meta_model(model, n_epoch, best_loss, args)
                best_epoch_results = results

            else:
                epochs_no_improve += 1
                print("no improvement since {} epoch".format(epochs_no_improve))
                if epochs_no_improve >= args.patience:
                    print(
                        "Earlystopping ({} epochs without improvement)".format(
                            args.patience
                        )
                    )
                    print("\nbest result : ", best_epoch_results)
                    break
            t_after_evalsaving = time()
            t_total_evalsaving += t_after_evalsaving - t_before_evalsaving
            print("KK timers : ", round(t_total_train, 4), round(t_total_eval, 4), round(t_total_evalsaving, 4))
            

        total_timer_end = datetime.now()
        total_time_elapsed = total_timer_end - total_timer_start

        print("Training ended. Total time elapsed = {}".format(total_time_elapsed))
