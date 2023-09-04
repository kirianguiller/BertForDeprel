import json
import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.mps
from conllup.conllup import emptyNodeJson, emptySentenceJson, sentenceJson_T
from torch.utils.data import DataLoader

from BertForDeprel.parser.cmds.predict import Predictor
from BertForDeprel.parser.cmds.train import Trainer
from BertForDeprel.parser.modules.BertForDepRel import (
    BertForDeprel,
    DataDescription,
    EvalResult,
)
from BertForDeprel.parser.utils.annotation_schema import compute_annotation_schema
from BertForDeprel.parser.utils.gpu_utils import get_devices_configuration
from BertForDeprel.parser.utils.load_data_utils import load_conllu_sentences
from BertForDeprel.parser.utils.types import (
    ModelParams_T,
    PredictionConfig,
    TrainingConfig,
)

PARENT = Path(__file__).parent
PATH_TEST_DATA_FOLDER = PARENT / "data"
PATH_DIAGNOSTIC_OUTPUT_FOLDER = PATH_TEST_DATA_FOLDER / "diagnostics"
PATH_DIAGNOSTIC_OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
PATH_MODELS_DIR = PATH_TEST_DATA_FOLDER / "models"

PATH_TRAIN_NAIJA = PATH_TEST_DATA_FOLDER / "naija.train.conllu"
PATH_TEST_NAIJA = PATH_TEST_DATA_FOLDER / "naija.test.conllu"
PATH_EXPECTED_PREDICTIONS_NAIJA = (
    PATH_TEST_DATA_FOLDER / "naija.predictions.expected.json"
)
NAIJA_MODEL_DIR = PATH_MODELS_DIR / "naija"

PATH_TEST_ENGLISH = PATH_TEST_DATA_FOLDER / "english.test.conllu"
PATH_TRAIN_ENGLISH = PATH_TEST_DATA_FOLDER / "english.train.conllu"
PATH_EXPECTED_PREDICTIONS_ENGLISH = (
    PATH_TEST_DATA_FOLDER / "english.predictions.expected.json"
)
ENGLISH_MODEL_DIR = PATH_MODELS_DIR / "english"
SEED = 42
DEVICE_CONFIG = get_devices_configuration("-1")

# not needed, but can boost confidence when debugging
# torch.use_deterministic_algorithms(True)


# generated model is used in other tests
@pytest.mark.order("first")
@pytest.mark.slow
@pytest.mark.fragile
def test_train_naija_model():
    _test_model_train_single(
        PATH_TRAIN_NAIJA,
        PATH_TEST_NAIJA,
        NAIJA_MODEL_DIR,
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
                data_description=DataDescription(0, 0, 0, 0),
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
                data_description=DataDescription(
                    n_train_sents=39,
                    n_test_sents=5,
                    n_train_batches=3,
                    n_test_batches=1,
                ),
            ),
        ],
    )


# generated model is used in other tests
@pytest.mark.order("first")
@pytest.mark.slow
@pytest.mark.fragile
def test_train_english_model():
    _test_model_train_single(
        PATH_TRAIN_ENGLISH,
        PATH_TEST_ENGLISH,
        ENGLISH_MODEL_DIR,
        [
            EvalResult(
                LAS_epoch=0.0,
                LAS_chuliu_epoch=0.0,
                acc_head_epoch=0.08,
                acc_deprel_epoch=0.008,
                acc_uposs_epoch=0.088,
                acc_xposs_epoch=1.0,
                acc_feats_epoch=0.008,
                acc_lemma_scripts_epoch=0.008,
                loss_head_epoch=0.322,
                loss_deprel_epoch=0.318,
                loss_uposs_epoch=0.258,
                loss_xposs_epoch=0.037,
                loss_feats_epoch=0.341,
                loss_lemma_scripts_epoch=0.298,
                loss_epoch=0.262,
                data_description=DataDescription(0, 0, 0, 0),
            ),
            EvalResult(
                LAS_epoch=0.008,
                LAS_chuliu_epoch=0.008,
                acc_head_epoch=0.088,
                acc_deprel_epoch=0.28,
                acc_uposs_epoch=0.104,
                acc_xposs_epoch=1.0,
                acc_feats_epoch=0.008,
                acc_lemma_scripts_epoch=0.144,
                loss_head_epoch=0.321,
                loss_deprel_epoch=0.295,
                loss_uposs_epoch=0.251,
                loss_xposs_epoch=0.026,
                loss_feats_epoch=0.332,
                loss_lemma_scripts_epoch=0.268,
                loss_epoch=0.249,
                data_description=DataDescription(
                    n_train_sents=50,
                    n_test_sents=10,
                    n_train_batches=4,
                    n_test_batches=1,
                ),
            ),
        ],
    )


def _test_model_train_single(path_train, path_test, path_out, expected_eval):
    # for reproducibility we need to set the seed during training; for example,
    # nn.Dropout uses rng at training time to drop a layer's weights, but at test
    # time it doesn't drop anything, so there's no random behavior then.
    torch.manual_seed(SEED)
    train_sentences = list(load_conllu_sentences(path_train))
    annotation_schema = compute_annotation_schema(train_sentences)
    model = BertForDeprel.new_model(
        "xlm-roberta-large", annotation_schema, DEVICE_CONFIG.device
    )
    test_sentences = load_conllu_sentences(path_test)
    training_config = TrainingConfig(
        max_epochs=1,
        patience=0,
    )
    trainer = Trainer(
        training_config,
        DEVICE_CONFIG.multi_gpu,
    )
    scores_generator = trainer.train(model, train_sentences, test_sentences)
    scores = [next(scores_generator), next(scores_generator)]
    scores = [s.rounded(3) for s in scores]
    model.save(  # type: ignore https://github.com/pytorch/pytorch/issues/81462
        path_out, training_config
    )

    assert scores == pytest.approx(expected_eval)


@pytest.mark.slow
@pytest.mark.fragile
def test_predict_multilingual():
    model_config = ModelParams_T.from_model_path(NAIJA_MODEL_DIR)

    model = BertForDeprel.load_pretrained_for_prediction(
        {"naija": NAIJA_MODEL_DIR, "english": ENGLISH_MODEL_DIR},
        "naija",
        DEVICE_CONFIG.device,
    )
    predictor = Predictor(
        model,
        PredictionConfig(batch_size=model_config.batch_size, num_workers=1),
        False,
    )

    naija_sentences = list(load_conllu_sentences(PATH_TEST_NAIJA))
    # add a sentence too large for the model; this should be skipped in the output
    too_long = emptySentenceJson()
    too_long["metaJson"]["sent_id"] = "too_long"
    for i in range(model.max_position_embeddings + 10):
        too_long["treeJson"]["nodesJson"][f"{i}"] = emptyNodeJson(ID=f"{i}")
    naija_sentences.insert(2, too_long)

    # On my M1, it's <7s.
    _test_predict_single(
        predictor, naija_sentences, PATH_EXPECTED_PREDICTIONS_NAIJA, 10
    )

    english_sentences = list(load_conllu_sentences(PATH_TEST_ENGLISH))
    model.activate("english")
    _test_predict_single(
        predictor, english_sentences, PATH_EXPECTED_PREDICTIONS_ENGLISH, 10
    )


predict_id = 0


def _test_predict_single(
    predictor: Predictor, input: List[sentenceJson_T], expected: Path, max_seconds: int
):
    actual, elapsed_seconds = predictor.predict(input)

    with open(expected, "r") as f:
        expected = json.load(f)

    global predict_id
    predict_id += 1
    with (PATH_DIAGNOSTIC_OUTPUT_FOLDER / f"actual-{predict_id}.json").open("w") as f:
        json.dump(actual, f, indent=2)

    assert actual == expected
    # On my M2, it's <7s.
    if elapsed_seconds > max_seconds:
        print(
            f"WARNING: Prediction took a long time: {elapsed_seconds} seconds.",
            file=sys.stderr,
        )


@pytest.mark.slow
@pytest.mark.fragile
def test_eval():
    """There is no eval API, per se, but this demonstrates how to do it. TODO: it's
    pretty convoluted."""
    model = BertForDeprel.load_single_pretrained_for_prediction(
        NAIJA_MODEL_DIR, DEVICE_CONFIG.device
    )

    sentences = load_conllu_sentences(PATH_TEST_NAIJA)
    test_dataset = model.encode_dataset(sentences)
    test_loader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_train,
        batch_size=16,
        num_workers=1,
    )

    results = model.eval_on_dataset(test_loader)

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
