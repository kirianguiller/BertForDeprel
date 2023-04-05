import os
from pathlib import Path

import pytest
import torch

from BertForDeprel.parser.utils.load_data_utils import ConlluDataset
from BertForDeprel.parser.utils.types import ModelParams_T, get_empty_annotation_schema

PATH_TEST_DATA_FOLDER = Path(__file__).parent / "data"
PATH_TEST_MODELS_FOLDER = Path(__file__).parent / "models"
PATH_TEST_CONLLU = str(PATH_TEST_DATA_FOLDER / "english.conllu")

model_params_test: ModelParams_T = {
    "model_folder_path": str(PATH_TEST_MODELS_FOLDER),
    "annotation_schema": get_empty_annotation_schema(),
    "max_epoch": 5,
    "patience": 3,
    "batch_size": 4,
    "maxlen": 512,
    "embedding_type": "bert-base-uncased",
    "embedding_cached_path": str(PATH_TEST_MODELS_FOLDER),
}


def test_health():
    assert True


def test_is_there_test_conllu():
    assert os.path.isfile(PATH_TEST_CONLLU)


def test_create_instance_dataset():
    dataset = ConlluDataset(PATH_TEST_CONLLU, model_params_test.copy(), "train",
                            compute_annotation_schema_if_not_found=True)
    assert len(dataset.sequences) == 2


def test_raise_error_if_no_annotation_schema():
    with pytest.raises(Exception):
        dataset = ConlluDataset(PATH_TEST_CONLLU, model_params_test.copy(), "train")


def test_predict_output():
    dataset = ConlluDataset(PATH_TEST_CONLLU, model_params_test, "predict", compute_annotation_schema_if_not_found=True)
    assert dataset[0]["idx"] == 0
    assert dataset[0]["seq_ids"] == [101, 2054, 2003, 1996, 5700, 3462, 2013, 3731, 2000, 1038, 9148, 2008, 4240, 1037,
                                     19782, 102]
    assert dataset[0]["attn_masks"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert dataset[0]["idx_convertor"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
    assert dataset[0]["subwords_start"] == [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
    assert dataset[0]["tokens_len"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]


def test_train_output():
    dataset = ConlluDataset(PATH_TEST_CONLLU, model_params_test, "train", compute_annotation_schema_if_not_found=True)
    assert dataset[0]["idx"] == 0
    assert dataset[0]["uposs"] == [-1, 6, 2, 3, 0, 4, 1, 7, 1, 7, -1, 1, 8, 3, 4]
    assert dataset[0]["heads"] == [-1, 2, 0, 5, 5, 2, 5, 6, 5, 8, -1, 5, 11, 14, 12]
    assert dataset[0]["deprels"] == [-1, 2, 8, 4, 6, 9, 10, 1, 10, 1, -1, 7, 1, 4, 1]


def test_collate_fn():
    dataset = ConlluDataset(PATH_TEST_CONLLU, model_params_test, "train", compute_annotation_schema_if_not_found=True)
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    assert torch.equal(batch["deprels"], torch.tensor([[-1,  2,  8,  4,  6,  9, 10,  1, 10,  1, -1,  7,  1,  4,  1, -1],
                                                       [-1,  2,  8,  3,  9,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]))

def test_add_prediction_to_sentence_json():
    dataset = ConlluDataset(PATH_TEST_CONLLU, model_params_test, "train", compute_annotation_schema_if_not_found=True)
    predicted_sentence_json =  dataset.add_prediction_to_sentence_json(0, [2,3,4,2,5], [0,0,0,0,0], [1,2,4,15,4], [5,2,3,4,3], [2,3,4,2,5], [5,2,3,4,3])
    assert predicted_sentence_json["treeJson"]["nodesJson"]["4"]["HEAD"] == 15


def test_get_contrained_dependency_for_chuliu():
    dataset = ConlluDataset(PATH_TEST_CONLLU, model_params_test, "test", compute_annotation_schema_if_not_found=True)
    assert dataset.get_contrained_dependency_for_chuliu(1) == [(1, 2), (2, 0), (3, 4), (4, 2), (5, 4)]
