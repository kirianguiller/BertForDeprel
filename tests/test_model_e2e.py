import json
import sys
from pathlib import Path

import pytest
import torch

from BertForDeprel.parser.cmds.predict import Predictor
from BertForDeprel.parser.cmds.train import Trainer
from BertForDeprel.parser.utils.gpu_utils import get_devices_configuration
from BertForDeprel.parser.utils.load_data_utils import ConlluDataset
from BertForDeprel.parser.utils.types import AnnotationSchema_T, ModelParams_T

PARENT = Path(__file__).parent
PATH_TEST_DATA_FOLDER = PARENT / "data"

PATH_TEST_CONLLU = PATH_TEST_DATA_FOLDER / "naija.test.conllu"
PATH_TRAIN_CONLLU = PATH_TEST_DATA_FOLDER / "naija.train.conllu"
PATH_EXPECTED_PREDICTIONS = PATH_TEST_DATA_FOLDER / "naija.predictions.expected.json"
PATH_MODELS_DIR = PATH_TEST_DATA_FOLDER / "models"


def _test_model_train():
    torch.manual_seed(42)
    model_config = ModelParams_T(
        model_folder_path=str(PATH_MODELS_DIR),
        max_epoch=1,
        patience=0,
        batch_size=16,
        max_position_embeddings=512,
        embedding_type="xlm-roberta-large",
    )

    device, _, multi_gpu = get_devices_configuration("-1")
    train_dataset = ConlluDataset(
        str(PATH_TRAIN_CONLLU),
        model_config,
        "train",
        compute_annotation_schema_if_not_found=True,
    )
    test_dataset = ConlluDataset(str(PATH_TEST_CONLLU), model_config, "train")
    # TODO: we have to create train_dataset before calling Trainer()
    # because the former sets the annotation schema in our model config,
    # and the latter uses it to set up the model (reversing the ordering results in
    # errors related to length-0 tensors). That is really clumsy and annoying.
    trainer = Trainer(
        model_config, device, multi_gpu, overwrite_pretrain_classifiers=False
    )

    scores = list(trainer.train(train_dataset, test_dataset))

    # TODO: this API should take the output path as an argument
    trainer.model.save_model(1)  # type: ignore https://github.com/pytorch/pytorch/issues/81462 # noqa: E501

    # TODO: put time in result and check that, as well; or specify deadline to pytest
    # TODO: these numbers are different on every machine, and therefore this test FAILS
    # anywhere except for mine. Need to figure out how to make it pass anywhere.
    assert scores == pytest.approx(
        [
            {
                "LAS_epoch": 0.0,
                "LAS_chuliu_epoch": 0.0,
                "acc_head_epoch": 0.077,
                "acc_deprel_epoch": 0.0,
                "acc_uposs_epoch": 0.046,
                "acc_xposs_epoch": 1.0,
                "acc_feats_epoch": 0.0,
                "acc_lemma_scripts_epoch": 0.0,
                "loss_head_epoch": 3.045,
                "loss_deprel_epoch": 3.611,
                "loss_xposs_epoch": 0.531,
                "loss_feats_epoch": 3.367,
                "loss_lemma_scripts_epoch": 3.452,
                "loss_epoch": 17.016,
                "n_sentences_train": 39,
                "n_sentences_test": 5,
                "epoch": 0,
            },
            {
                "LAS_epoch": 0.015,
                "LAS_chuliu_epoch": 0.015,
                "acc_head_epoch": 0.123,
                "acc_deprel_epoch": 0.308,
                "acc_uposs_epoch": 0.046,
                "acc_xposs_epoch": 1.0,
                "acc_feats_epoch": 0.0,
                "acc_lemma_scripts_epoch": 0.0,
                "loss_head_epoch": 3.041,
                "loss_deprel_epoch": 3.37,
                "loss_xposs_epoch": 0.415,
                "loss_feats_epoch": 3.228,
                "loss_lemma_scripts_epoch": 3.133,
                "loss_epoch": 16.117,
                "n_sentences_train": 39,
                "n_sentences_test": 5,
                "epoch": 1,
            },
        ]
    )


def _test_predict():
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
    # _test_model_train()
    _test_predict()
