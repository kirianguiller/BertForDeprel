from pathlib import Path

import torch

from BertForDeprel.parser.utils.annotation_schema import compute_annotation_schema
from BertForDeprel.parser.utils.load_data_utils import load_conllu_sentences
from BertForDeprel.parser.utils.types import ModelParams_T
from BertForDeprel.parser.utils.ud_dataset import PartialPredictionConfig, UDDataset

PATH_TEST_DATA_FOLDER = Path(__file__).parent / "data"
PATH_TEST_MODELS_FOLDER = Path(__file__).parent / "models"
PATH_TEST_CONLLU = PATH_TEST_DATA_FOLDER / "english.conllu"

model_params_test = ModelParams_T(
    max_epoch=5,
    patience=3,
    batch_size=4,
    max_position_embeddings=512,
    embedding_type="bert-base-uncased",
)


def get_test_instance():
    sentences = load_conllu_sentences(PATH_TEST_CONLLU)
    annotation_schema = compute_annotation_schema(sentences)

    return UDDataset(
        sentences,
        annotation_schema,
        model_params_test.embedding_type,
        model_params_test.max_position_embeddings,
    )


def test_create_instance_dataset():
    dataset = get_test_instance()
    assert len(dataset.sequences) == 2


def test_predict_output():
    dataset = get_test_instance()
    assert dataset[0].idx == 0
    assert dataset[0].sequence_token_ids == [
        101,
        2054,
        2003,
        1996,
        5700,
        3462,
        2013,
        3731,
        2000,
        1038,
        9148,
        2008,
        4240,
        1037,
        19782,
        102,
    ]
    assert dataset[0].tok_starts_word == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
    ]
    assert dataset[0].idx_converter == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
    assert dataset[0].tokens_len == [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]


def test_train_output():
    dataset = get_test_instance()
    assert dataset[0].idx == 0
    assert dataset[0].uposs == [-1, 6, 2, 3, 0, 4, 1, 7, 1, 7, -1, 1, 8, 3, 4]
    assert dataset[0].heads == [-1, 2, 0, 5, 5, 2, 5, 6, 5, 8, -1, 5, 11, 14, 12]
    assert dataset[0].deprels == [-1, 2, 8, 4, 6, 9, 10, 1, 10, 1, -1, 7, 1, 4, 1]


def test_collate_train_fn():
    dataset = get_test_instance()
    batch = dataset.collate_train([dataset[0], dataset[1]])
    assert torch.equal(
        batch.deprels,
        torch.tensor(
            [
                [-1, 2, 8, 4, 6, 9, 10, 1, 10, 1, -1, 7, 1, 4, 1, -1],
                [-1, 2, 8, 3, 9, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        ),
    )
    assert torch.equal(
        batch.attn_masks,
        torch.tensor(
            [
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )


def test_add_prediction_to_sentence_json_keep_none():
    dataset = get_test_instance()

    # Check for keep_* = NONE
    predicted_sentence_json_none = dataset.construct_sentence_prediction(
        1,
        uposs_preds=[2, 3, 4, 2, 5],
        xposs_preds=[0, 0, 0, 0, 0],
        chuliu_heads=[1, 2, 4, 15, 4],
        deprels_pred_chulius=[5, 2, 3, 4, 3],
        feats_preds=[2, 3, 4, 2, 5],
        lemma_scripts_preds=[5, 2, 3, 4, 3],
        partial_pred_config=PartialPredictionConfig(),
    )
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["1"]["HEAD"] == 1
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["2"]["HEAD"] == 2
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["3"]["HEAD"] == 4
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["4"]["HEAD"] == 15
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["5"]["HEAD"] == 4

    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["1"]["UPOS"] == "AUX"
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["2"]["UPOS"] == "DET"
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["3"]["UPOS"] == "NOUN"
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["4"]["UPOS"] == "AUX"
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["5"]["UPOS"] == "NUM"

    assert (
        predicted_sentence_json_none["treeJson"]["nodesJson"]["1"]["DEPREL"] == "flat"
    )
    assert (
        predicted_sentence_json_none["treeJson"]["nodesJson"]["2"]["DEPREL"]
        == "comp:pred"
    )
    assert (
        predicted_sentence_json_none["treeJson"]["nodesJson"]["3"]["DEPREL"]
        == "compound"
    )
    assert predicted_sentence_json_none["treeJson"]["nodesJson"]["4"]["DEPREL"] == "det"
    assert (
        predicted_sentence_json_none["treeJson"]["nodesJson"]["5"]["DEPREL"]
        == "compound"
    )


def test_add_prediction_to_sentence_json_keep_existing():
    dataset = get_test_instance()

    # Check for keep_* = EXISTING
    predicted_sentence_json_existing = dataset.construct_sentence_prediction(
        1,
        uposs_preds=[2, 3, 4, 2, 5],
        xposs_preds=[0, 0, 0, 0, 0],
        chuliu_heads=[1, 2, 4, 15, 4],
        deprels_pred_chulius=[5, 2, 3, 4, 3],
        feats_preds=[2, 3, 4, 2, 5],
        lemma_scripts_preds=[5, 2, 3, 4, 3],
        partial_pred_config=PartialPredictionConfig(
            keep_upos="EXISTING",
            keep_xpos="EXISTING",
            keep_deprels="EXISTING",
            keep_feats="EXISTING",
            keep_heads="EXISTING",
            keep_lemmas="EXISTING",
        ),
    )
    assert predicted_sentence_json_existing["treeJson"]["nodesJson"]["1"]["HEAD"] == 2
    assert predicted_sentence_json_existing["treeJson"]["nodesJson"]["2"]["HEAD"] == 0
    assert predicted_sentence_json_existing["treeJson"]["nodesJson"]["3"]["HEAD"] == 4
    assert predicted_sentence_json_existing["treeJson"]["nodesJson"]["4"]["HEAD"] == 2
    assert predicted_sentence_json_existing["treeJson"]["nodesJson"]["5"]["HEAD"] == 4

    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["1"]["UPOS"] == "PRON"
    )
    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["2"]["UPOS"] == "AUX"
    )
    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["3"]["UPOS"] == "NOUN"
    )
    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["4"]["UPOS"]
        == "PROPN"
    )
    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["5"]["UPOS"] == "NUM"
    )

    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["1"]["DEPREL"]
        == "comp:pred"
    )
    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["2"]["DEPREL"]
        == "root"
    )
    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["3"]["DEPREL"]
        == "compound"
    )
    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["4"]["DEPREL"]
        == "subj"
    )
    assert (
        predicted_sentence_json_existing["treeJson"]["nodesJson"]["5"]["DEPREL"]
        == "flat"
    )


def test_add_prediction_to_sentence_json_keep_all():
    dataset = get_test_instance()
    # Check for keep_* = ALL
    predicted_sentence_json_all = dataset.construct_sentence_prediction(
        1,
        uposs_preds=[2, 3, 4, 2, 5],
        xposs_preds=[0, 0, 0, 0, 0],
        chuliu_heads=[1, 2, 4, 15, 4],
        deprels_pred_chulius=[5, 2, 3, 4, 3],
        feats_preds=[2, 3, 4, 2, 5],
        lemma_scripts_preds=[5, 2, 3, 4, 3],
        partial_pred_config=PartialPredictionConfig(
            keep_upos="ALL",
            keep_xpos="ALL",
            keep_deprels="ALL",
            keep_feats="ALL",
            keep_heads="ALL",
            keep_lemmas="ALL",
        ),
    )
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["1"]["HEAD"] == 2
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["2"]["HEAD"] == 0
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["3"]["HEAD"] == 4
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["4"]["HEAD"] == 2
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["5"]["HEAD"] == 4

    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["1"]["UPOS"] == "PRON"
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["2"]["UPOS"] == "AUX"
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["3"]["UPOS"] == "NOUN"
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["4"]["UPOS"] == "PROPN"
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["5"]["UPOS"] == "NUM"

    assert (
        predicted_sentence_json_all["treeJson"]["nodesJson"]["1"]["DEPREL"]
        == "comp:pred"
    )
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["2"]["DEPREL"] == "root"
    assert (
        predicted_sentence_json_all["treeJson"]["nodesJson"]["3"]["DEPREL"]
        == "compound"
    )
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["4"]["DEPREL"] == "subj"
    assert predicted_sentence_json_all["treeJson"]["nodesJson"]["5"]["DEPREL"] == "flat"


def test_get_contrained_dependency_for_chuliu():
    dataset = get_test_instance()
    assert dataset.get_constrained_dependency_for_chuliu(1) == [
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 2),
        (5, 4),
    ]
