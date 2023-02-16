import os
from BertForDeprel.parser.utils.load_data_utils import ConlluDataset
from BertForDeprel.parser.utils.types import ModelParams_T, get_empty_annotation_schema
from pathlib import Path


PATH_TEST_DATA_FOLDER = Path(__file__).parent / "data"
PATH_TEST_MODELS_FOLDER = Path(__file__).parent / "models"
PATH_TEST_CONLLU = PATH_TEST_DATA_FOLDER / "english.conllu"

model_params_test: ModelParams_T = {
    "root_folder_path": PATH_TEST_MODELS_FOLDER,
    "model_name": "test_model",
    "annotation_schema": get_empty_annotation_schema(),
    "max_epoch": 5,
    "patience": 3,
    "batch_size": 4,
    "maxlen": 512,
    "embedding_type": "bert-base-uncased",
    "embedding_cached_path": PATH_TEST_MODELS_FOLDER,
}

def test_health():
    assert True

def test_is_there_test_conllu():
    assert os.path.isfile(PATH_TEST_CONLLU)

# def test_create_instance_dataset():
#     dataset = ConlluDataset(PATH_TEST_CONLLU, "bert-base-cased", {})
