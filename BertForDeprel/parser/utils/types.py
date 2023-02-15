from typing import TypedDict, List, Dict


class EpochResults_T(TypedDict):
    LAS: float
    LAS_chuliu: float
    acc_deprel: float
    acc_pos: float
    loss_head: float
    loss_deprel: float
    loss_total: float


class AnnotationSchema_T(TypedDict):
    deprels: List[str]
    uposs: List[str]

    dep2i: Dict[str, int]
    i2dep: Dict[int, str]    
    pos2i: Dict[str, int]
    i2pos: Dict[int, str]


class ModelParams_T(TypedDict):
    # Shared
    root_folder_path: str
    model_name: str
    annotation_schema: AnnotationSchema_T

    # Next training params (only relevent if one want to train a model or retrain/finetune)
    max_epoch: int
    patience: int
    batch_size: int
    maxlen: int

    # Embedding (xlm-roberta-large / bert-multilingual-base-uncased ...)
    embedding_type: str
    embedding_cached_path: str

    # Finetuned training meta params
    # n_current_epoch: int
    # current_epoch_results: EpochResults_T


def get_empty_current_epoch_results() -> EpochResults_T:
    return {
        "LAS": -1,
        "LAS_chuliu": -1,
        "acc_deprel": -1,
        "acc_pos": -1,
        "loss_head": -1,
        "loss_deprel": -1,
        "loss_total": -1,
    }


def get_empty_annotation_schema() -> AnnotationSchema_T:
    return {
        "deprels": [],
        "uposs": [],

        "dep2i": {},
        "i2dep": {},
        "pos2i": {},
        "i2pos": {},
    }


def get_default_model_params() -> ModelParams_T:
    params: ModelParams_T = {
        "root_folder_path": "",
        "model_name": "my_bert_for_deprel_model",
        "annotation_schema": get_empty_annotation_schema(),
        "max_epoch": 30,
        "patience": 5,
        "batch_size": 8,
        "maxlen": 512,
        "embedding_type": "xlm-roberta-large",
        "embedding_cached_path": "",
        # "n_current_epoch": 0,
        # "current_epoch_results": get_empty_current_epoch_results(),
    }
    return params
