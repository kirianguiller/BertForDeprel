# -*- coding: utf-8 -*-
import os
from datetime import datetime
from collections import OrderedDict
from parser import BertForDeprel
from parser.cmds.cmd import CMD
from parser.utils.load_data_utils import ConlluDataset
from parser.utils.train_utils import eval_epoch
from parser.utils.os_utils import path_or_name


import torch
from torch import nn
from torch.utils.data import DataLoader



class Evaluate(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--batch_size', default=32, type=int,
                               help='batch size')
        # subparser.add_argument('--buckets', default=32, type=int,
        #                        help='max num of buckets to use')
        subparser.add_argument('--punct', action='store_true',
                               help='whether to include punctuation')
        subparser.add_argument('--feval', default='',
                               help='path to dataset')

        return subparser

    def __call__(self, args):
        super(Evaluate, self).__call__(args)
        if path_or_name(args.feval) == "name":
            args.feval = os.path.join(args.folder, 'eval', args.feval)
        # if not args.feval:
        #     path_eval_folder = os.path.join(args.folder, "eval")
        #     name_eval_file = os.listdir(path_eval_folder)[0]
        #     args.feval = os.path.join(path_eval_folder, name_eval_file)
        #     print("path to feval (default behavior) :", args.feval)

        print("Load the saved config")
        checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
        loaded_args = checkpoint['args']

        print("Load the dataset")
        self.load_tokenizer(loaded_args)
        eval_dataset = ConlluDataset(args.feval, self.tokenizer, loaded_args)
        params = {
          "batch_size": args.batch_size, 
          "num_workers": args.num_workers,
        }
        print(args.num_workers)
        print(type(args.num_workers))
        eval_loader = DataLoader(eval_dataset, **params)

        print(f"{'eval:':6} {len(eval_dataset):5} sentences, "
          f"{len(eval_loader):3} batches, ")


        print("Load the model")
        model = BertForDeprel(loaded_args)
        model.to(args.device)

        if args.multi_gpu:
            print("MODEL TO MULTI GPU")
            model = nn.DataParallel(model)

        ### To reactivate if probleme in the loading of the model states
        # loaded_state_dict = OrderedDict()
        # for k, v in checkpoint['state_dict'].items():
        #     name = k.replace("module.","")
        #     loaded_state_dict[name] = v
        
        # model.load_state_dict(loaded_state_dict)

        
        model.load_state_dict(checkpoint['state_dict'])

        
        results = eval_epoch(model, eval_loader, loaded_args)

        print(results)