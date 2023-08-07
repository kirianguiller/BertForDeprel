import json
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch

from BertForDeprel.parser.cmds.predict import Predictor
from BertForDeprel.parser.cmds.train import Train
from BertForDeprel.parser.utils.load_data_utils import ConlluDataset
from BertForDeprel.parser.utils.types import AnnotationSchema_T, ModelParams_T

PARENT = Path(__file__).parent
PATH_TEST_DATA_FOLDER = PARENT / "data"

PATH_TEST_CONLLU = PATH_TEST_DATA_FOLDER / "naija.test.conllu"
PATH_TRAIN_CONLLU = PATH_TEST_DATA_FOLDER / "naija.train.conllu"
PATH_EXPECTED_PREDICTIONS = PATH_TEST_DATA_FOLDER / "naija.predictions.expected.json"
PATH_MODELS_DIR = PATH_TEST_DATA_FOLDER / "models"


@dataclass
class TrainArgs(Namespace):
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
        model_folder_path=str(PATH_MODELS_DIR),
        max_epoch=1,
        patience=0,
        batch_size=16,
        max_position_embeddings=512,
        embedding_type="xlm-roberta-large",
    )

    train = Train()
    # TODO: should return scores, timing, model config, and model output path
    # yield timing, epoch number, scores, current model and current best scores and
    # model for each epoch; return total timing, epochs, and best scores and model
    # TODO: create simpler API
    train.run(train_args, model_config)
    with open(model_config.model_folder_path + "/scores.best.json", "r") as f:
        scores = json.load(f)
    # TODO: put time in result and check that, as well; or specify it to pytest somehow
    # TODO: these numbers are different on every machine, and therefore this test
    # FAILS anywhere except for mine. Need to figure out how to make it pass anywhere.
    assert scores == pytest.approx(
        {
            "LAS_epoch": 0.046,
            "LAS_chuliu_epoch": 0.046,
            "acc_head_epoch": 0.154,
            "acc_deprel_epoch": 0.308,
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


def _test_predict():
    # TODO: Next: figure out why this is failing.
    model_config = ModelParams_T()
    with open(PATH_MODELS_DIR / "config.json", "r") as f:
        config_dict = json.load(f)
        model_config.__dict__.update(config_dict)
        annotation_schema = AnnotationSchema_T()
        annotation_schema.__dict__.update(config_dict["annotation_schema"])
        model_config.annotation_schema = annotation_schema

    predictor = Predictor(model_config, 1)
    # TODO: don't pass full model config; just annotation_schema and
    # max_position_embeddings
    pred_dataset = ConlluDataset(str(PATH_TEST_CONLLU), model_config, "predict")
    actual, elapsed_seconds = predictor.predict(pred_dataset)

    with open(PATH_EXPECTED_PREDICTIONS, "r") as f:
        expected = json.load(f)

    assert actual == expected
    # On my M2, it's <7s.
    if elapsed_seconds > 10:
        print(
            f"WARNING: Prediction took a long time: {elapsed_seconds} seconds.",
            file=sys.stderr,
        )


# About 30s on my M2 Mac.
@pytest.mark.slow
@pytest.mark.fragile
def test_train_and_predict():
    _test_model_train()
    _test_predict()
