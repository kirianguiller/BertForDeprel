# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
from ..utils.types import ModelParams_T


class CMD(object):
    def __call__(self, args, model_params: ModelParams_T):
        self.args = args

