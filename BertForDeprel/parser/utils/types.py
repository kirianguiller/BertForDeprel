import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from .annotation_schema import AnnotationSchema_T

CONFIG_FILE_NAME = "config.json"
MODEL_FILE_NAME = "model.pt"


@dataclass
class ModelParams_T:
    # Shared
    # Pre-trained embeddings to download from 🤗 (xlm-roberta-large /
    # bert-multilingual-base-uncased ...)
    embedding_type: str = "xlm-roberta-large"
    annotation_schema: AnnotationSchema_T = field(default_factory=AnnotationSchema_T)
    # how many subprocesses to use for data loading. 0 means that the data will be
    # loaded in the main process. (default: 0)
    num_workers: int = 1
    # How many sentences to process in each batch
    batch_size: int = 16

    # Training params
    # In our experiments, most of the models based on UD data converged in 10-15 epochs
    max_epoch: int = 15
    # How many epochs with no performance improvement before training is ended early
    patience: int = 3

    @staticmethod
    def from_model_path(model_path: Path) -> "ModelParams_T":
        """Load model parameters from the model directory."""
        with open(model_path / CONFIG_FILE_NAME, "r") as f:
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
class PredictionConfig:
    # How many sentences to process in each batch
    batch_size: int = 16
    # how many subprocesses to use for data loading
    num_workers: int = 1


@dataclass
class TrainingConfig(PredictionConfig):
    # In our experiments, most of the models based on UD data converged in 10-15 epochs
    max_epochs: int = 15
    # How many epochs with no performance improvement before training is ended early
    patience: int = 3
    # Number of digits to round the metrics to
    ndigits: int = 3
