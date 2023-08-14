import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from BertForDeprel.parser.cmds.predict import Predictor
from BertForDeprel.parser.cmds.train import Trainer
from BertForDeprel.parser.modules.BertForDepRel import BertForDeprel, EvalResult
from BertForDeprel.parser.utils.annotation_schema import compute_annotation_schema
from BertForDeprel.parser.utils.gpu_utils import get_devices_configuration
from BertForDeprel.parser.utils.load_data_utils import UDDataset, load_conllu_sentences
from BertForDeprel.parser.utils.types import (
    ModelParams_T,
    PredictionConfig,
    TrainingConfig,
)

PARENT = Path(__file__).parent
PATH_TEST_DATA_FOLDER = PARENT / "data"

PATH_TEST_CONLLU = PATH_TEST_DATA_FOLDER / "naija.test.conllu"
PATH_TRAIN_CONLLU = PATH_TEST_DATA_FOLDER / "naija.train.conllu"
PATH_EXPECTED_PREDICTIONS = PATH_TEST_DATA_FOLDER / "naija.predictions.expected.json"
PATH_MODELS_DIR = PATH_TEST_DATA_FOLDER / "models"


def _test_model_train():
    torch.manual_seed(42)

    train_sentences = load_conllu_sentences(PATH_TRAIN_CONLLU)
    annotation_schema = compute_annotation_schema(train_sentences)

    model = BertForDeprel.new_model("xlm-roberta-large", annotation_schema)

    device_config = get_devices_configuration("-1")

    train_dataset = UDDataset(
        train_sentences,
        model.annotation_schema,
        model.embedding_type,
        model.max_position_embeddings,
        "train",
    )

    test_sentences = load_conllu_sentences(PATH_TEST_CONLLU)
    test_dataset = UDDataset(
        test_sentences,
        model.annotation_schema,
        model.embedding_type,
        model.max_position_embeddings,
        "train",
    )
    training_config = TrainingConfig(
        max_epochs=1,
        patience=0,
    )
    trainer = Trainer(
        training_config,
        device_config,
    )

    scores_generator = trainer.train(model, train_dataset, test_dataset)
    scores = [next(scores_generator), next(scores_generator)]
    scores = [s.rounded(3) for s in scores]

    trainer.model.save_model(  # type: ignore https://github.com/pytorch/pytorch/issues/81462 # noqa: E501
        PATH_MODELS_DIR, training_config
    )

    # TODO: put time in result and check that, as well; or specify deadline to pytest
    # TODO: these numbers are different on every machine, and therefore this test FAILS
    # anywhere except for mine. Need to figure out how to make it pass anywhere.
    assert scores == pytest.approx(
        [
            EvalResult(
                LAS_epoch=0.0,
                LAS_chuliu_epoch=0.0,
                acc_head_epoch=0.077,
                acc_deprel_epoch=0.0,
                acc_uposs_epoch=0.046,
                acc_xposs_epoch=1.0,
                acc_feats_epoch=0.0,
                acc_lemma_scripts_epoch=0.0,
                loss_head_epoch=0.609,
                loss_deprel_epoch=0.722,
                loss_uposs_epoch=0.602,
                loss_xposs_epoch=0.106,
                loss_feats_epoch=0.673,
                loss_lemma_scripts_epoch=0.69,
                loss_epoch=0.567,
                training_diagnostics=None,
            ),
            EvalResult(
                LAS_epoch=0.015,
                LAS_chuliu_epoch=0.015,
                acc_head_epoch=0.123,
                acc_deprel_epoch=0.308,
                acc_uposs_epoch=0.046,
                acc_xposs_epoch=1.0,
                acc_feats_epoch=0.0,
                acc_lemma_scripts_epoch=0.0,
                loss_head_epoch=0.608,
                loss_deprel_epoch=0.674,
                loss_uposs_epoch=0.586,
                loss_xposs_epoch=0.083,
                loss_feats_epoch=0.646,
                loss_lemma_scripts_epoch=0.627,
                loss_epoch=0.537,
                training_diagnostics=None,
            ),
        ]
    )


def _test_predict():
    model_config = ModelParams_T.from_model_path(PATH_MODELS_DIR)
    device_config = get_devices_configuration("-1")

    model = BertForDeprel.load_pretrained_for_prediction(PATH_MODELS_DIR)
    predictor = Predictor(
        model,
        PredictionConfig(batch_size=model_config.batch_size, num_workers=1),
        device_config,
    )

    sentences = load_conllu_sentences(PATH_TEST_CONLLU)
    pred_dataset = UDDataset(
        sentences,
        model_config.annotation_schema,
        model_config.embedding_type,
        model_config.max_position_embeddings,
        "train",
    )

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


def _test_eval():
    """There is no eval API, per se, but this demonstrates how to do it. TODO: it's
    pretty convoluted."""
    model_config = ModelParams_T.from_model_path(PATH_MODELS_DIR)
    device_config = get_devices_configuration("-1")

    model = BertForDeprel.load_pretrained_for_prediction(PATH_MODELS_DIR)
    model = model.to(device_config.device)

    sentences = load_conllu_sentences(PATH_TEST_CONLLU)
    test_dataset = UDDataset(
        sentences,
        model_config.annotation_schema,
        model_config.embedding_type,
        model_config.max_position_embeddings,
        "train",
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn_train,
        batch_size=16,
        num_workers=1,
    )

    results = model.eval_on_dataset(test_loader, device_config.device)

    # TODO: these are different on each machine, and therefore this test FAILS anywhere
    # but mine.
    assert results.rounded(3) == pytest.approx(
        EvalResult(
            LAS_epoch=0.015,
            LAS_chuliu_epoch=0.015,
            acc_head_epoch=0.123,
            acc_deprel_epoch=0.308,
            acc_uposs_epoch=0.046,
            acc_xposs_epoch=1.0,
            acc_feats_epoch=0.0,
            acc_lemma_scripts_epoch=0.0,
            loss_head_epoch=0.608,
            loss_deprel_epoch=0.674,
            loss_uposs_epoch=0.586,
            loss_xposs_epoch=0.083,
            loss_feats_epoch=0.646,
            loss_lemma_scripts_epoch=0.627,
            loss_epoch=0.537,
            training_diagnostics=None,
        ),
    )


# About 30s on my M2 Mac.
@pytest.mark.slow
@pytest.mark.fragile
def test_train_and_predict():
    _test_model_train()
    _test_predict()
    _test_eval()
