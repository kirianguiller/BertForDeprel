# -*- coding: utf-8 -*-

import os
from parser import BertForDeprel #added

import torch
import torch.nn as nn
from transformers import BertTokenizer, CamembertTokenizer, AutoTokenizer



class CMD(object):

    def __call__(self, args):
        self.args = args

    def load_tokenizer(self, args):
        print("LOAD TOKENIZER")
        if args.bert_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif args.bert_type == "camembert":
            self.tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        elif args.bert_type == "mbert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert_type)

