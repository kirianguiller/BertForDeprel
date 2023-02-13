# -*- coding: utf-8 -*-

from transformers import AutoTokenizer


class CMD(object):

    def __call__(self, args):
        self.args = args

    def load_tokenizer(self, bert_type):
        print("LOAD TOKENIZER")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)

