import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from .annotation_schema import AnnotationSchema_T


@dataclass
class ModelParams_T:
    # Shared
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

    # Pre-trained embeddings to download from 🤗 (xlm-roberta-large /
    # bert-multilingual-base-uncased ...)
    embedding_type: str = "xlm-roberta-large"
    # Maximum length of an input sequence; the default value is the default from
    # xlm-roberta-large. Using larger values could result in doubling or quadrupling the
    # memory usage.
    max_position_embeddings: int = 512

    @staticmethod
    def from_model_path(model_path: Path) -> "ModelParams_T":
        """Load model parameters from the model directory."""
        with open(model_path / "config.json", "r") as f:
            params_dict = json.load(f)
        model_params = ModelParams_T()
        # TODO: Check the validity of this first; at least a version number
        model_params.__dict__.update(params_dict)
        if "annotation_schema" in params_dict:
            model_params.annotation_schema = AnnotationSchema_T.from_dict(
                params_dict["annotation_schema"]
            )

        return model_params


class DataclassJSONEncoder(json.JSONEncoder):
    """JSON encoder for data that may include dataclasses."""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, Path):
            return str(o)
        return super().default(o)


@dataclass
class TrainingConfig:
    # In our experiments, most of the models based on UD data converged in 10-15 epochs
    max_epochs: int = 15
    # How many epochs with no performance improvement before training is ended early
    patience: int = 3
    # How many sentences to process in each batch
    batch_size: int = 16
    # how many subprocesses to use for data loading
    num_workers: int = 1
    # Number of digits to round the metrics to
    ndigits: int = 3


@dataclass
class PredictionConfig:
    # How many sentences to process in each batch
    batch_size: int = 16
    # how many subprocesses to use for data loading
    num_workers: int = 1
