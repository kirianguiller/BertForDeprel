import json
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Optional

import pytest
import torch

from BertForDeprel.parser.cmds.predict import Predict
from BertForDeprel.parser.cmds.train import Train
from BertForDeprel.parser.utils.types import AnnotationSchema_T, ModelParams_T

PARENT = Path(__file__).parent
PATH_TEST_DATA_FOLDER = PARENT / "data"

# TODO: make dataset smaller so it can run in a reasonable amount of time
# (currently 2 minutes on MPS)
PATH_TEST_CONLLU = PATH_TEST_DATA_FOLDER / "naija.test.conllu"
PATH_TRAIN_CONLLU = PATH_TEST_DATA_FOLDER / "naija.train.conllu"
PATH_EXPECTED_PREDICTIONS = PATH_TEST_DATA_FOLDER / "naija.predictions.expected.conllu"


@dataclass
class TrainArgs:
    ftrain: str = str(PATH_TRAIN_CONLLU)
    ftest: str = str(PATH_TEST_CONLLU)
    num_workers: int = 1
    batch_size_eval: int = 16
    overwrite_pretrain_classifiers: bool = False
    mode: str = "train"
    # dummy stuff that must be defined to avoid errors
    model_folder_path: str = ""
    embedding_type: str = ""
    max_epoch: int = 0
    patience: int = 0
    path_annotation_schema: str = ""
    path_folder_compute_annotation_schema: str = ""
    conf_pretrain: str = ""
    device: Optional[torch.device] = None
    train_on_gpu: bool = False
    multi_gpu: bool = False


def _test_model_train():
    torch.manual_seed(42)
    train_args = TrainArgs()
    model_config = ModelParams_T(
        model_folder_path=str(PATH_TEST_DATA_FOLDER / "models"),
        max_epoch=1,
        patience=0,
        batch_size=16,
        max_position_embeddings=512,
        embedding_type="xlm-roberta-large",
    )

    train = Train()
    # TODO: should return scores, timing, model config, and model output path
    train(train_args, model_config)
    with open(model_config.model_folder_path + "/scores.best.json", "r") as f:
        scores = json.load(f)
    # TODO: put time in result and check that, as well; or specify it to pytest somehow
    # TODO: test *all* of the output scores
    assert scores == pytest.approx(
        {
            "LAS_epoch": 0.046,
            "LAS_chuliu_epoch": 0.046,
            "acc_head_epoch": 0.154,
            "acc_deprel_epoch": 0.3076923076923077,
            "acc_uposs_epoch": 0.062,
            "acc_xposs_epoch": 1.0,
            "acc_feats_epoch": 0.0,
            "acc_lemma_scripts_epoch": 0.0,
            "loss_head_epoch": 3.04,
            "loss_deprel_epoch": 3.37,
            "loss_xposs_epoch": 0.434,
            "loss_feats_epoch": 3.24,
            "loss_lemma_scripts_epoch": 3.185,
            "loss_epoch": 16.202,
            "n_sentences_train": 39,
            "n_sentences_test": 5,
            "epoch": 1,
        }
    )


class PredictArgs:
    def __init__(self):
        self.inpath = str(PATH_TEST_CONLLU)
        self.suffix = ".predicted"
        self.conf: ModelParams_T
        self.outpath: str
        self.overwrite = True
        self.keep_upos = False
        self.keep_xpos = False
        self.keep_feats = False
        self.keep_deprels = False
        self.keep_heads = False
        self.keep_lemmas = False
        self.num_workers = 1
        self.device = None
        self.train_on_gpu = False
        self.multi_gpu = False


def _test_predict():
    predict = Predict()
    model_config = ModelParams_T()
    with open(PATH_TEST_DATA_FOLDER / "models" / "config.json", "r") as f:
        config_dict = json.load(f)
        model_config.__dict__.update(config_dict)
        annotation_schema = AnnotationSchema_T()
        annotation_schema.__dict__.update(config_dict["annotation_schema"])
        model_config.annotation_schema = annotation_schema

    predict_args = PredictArgs()
    predict_args.conf = model_config
    predict_args.outpath = model_config.model_folder_path
    # TODO: should return output
    predict(predict_args, model_config)

    with open(PATH_EXPECTED_PREDICTIONS, "r") as f:
        expected_lines = f.readlines()
    with open(predict_args.outpath + "/naija.test.predicted.conllu", "r") as f:
        actual_lines = f.readlines()
    diff = list(unified_diff(expected_lines, actual_lines))
    assert diff == [], "Predicted ConllU lines differ from expected:" + "".join(diff)


# About 30s on my M2 Mac. Skip with -m "not slow".
@pytest.mark.slow
def test_train_and_predict():
    _test_model_train()
    _test_predict()
