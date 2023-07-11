from dataclasses import dataclass, field
import dataclasses
import json
from typing import List


@dataclass
class EpochResults_T:
    LAS: float
    LAS_chuliu: float
    acc_deprel: float
    acc_pos: float
    loss_head: float
    loss_deprel: float
    loss_total: float


@dataclass
class AnnotationSchema_T:
    deprels: List[str] = field(default_factory=list)
    uposs: List[str] = field(default_factory=list)
    xposs: List[str] = field(default_factory=list)
    feats: List[str] = field(default_factory=list)
    lemma_scripts: List[str] = field(default_factory=list)

@dataclass
class ModelParams_T:
    # Shared
    # TODO: what behavior does an empty path lead to?
    model_folder_path: str = ""
    # TODO: what behavior does an empty schema lead to?
    annotation_schema: AnnotationSchema_T = field(default_factory=AnnotationSchema_T)

    # Training params
    # In our experiments, most of the models based on UD data converged in 10-15 epochs
    max_epoch: int = 15
    # How many epochs with no performance improvement before training is ended early
    patience: int = 3
    # How many sentences to process in each batch
    batch_size: int = 16

    # Finetuned training meta params
    # n_current_epoch: int
    # current_epoch_results: EpochResults_T

    # Allows a copy command in the lemma scripts. In the UDPipe paper, they tried both with and
    # without this option and kept the one that yielded fewer unique scripts.
    allow_lemma_char_copy: bool = False

    # Pre-trained embeddings to download from 🤗 (xlm-roberta-large / bert-multilingual-base-uncased ...)
    embedding_type: str = "xlm-roberta-large"
    # Maximum length of an input sequence; the default value is the default from xlm-roberta-large.
    # Using larger values could result in doubling or quadrupling the memory usage.
    max_position_embeddings: int = 512


def get_empty_current_epoch_results():
    return EpochResults_T(
        LAS=-1,
        LAS_chuliu=-1,
        acc_deprel=-1,
        acc_pos=-1,
        loss_head=-1,
        loss_deprel=-1,
        loss_total=-1,
    )



class DataclassJSONEncoder(json.JSONEncoder):
    """JSON encoder for data that may include dataclasses."""
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
