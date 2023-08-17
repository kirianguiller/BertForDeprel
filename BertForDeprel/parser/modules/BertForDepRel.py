import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, Iterable, Optional, Self

import numpy as np
import torch
import torch.mps
from conllup.conllup import sentenceJson_T
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW
from transformers import (  # type: ignore (TODO: why can't PyLance find these?)
    AutoModel,
    XLMRobertaModel,
)
from transformers.adapters import PfeifferConfig

from ..utils.annotation_schema import AnnotationSchema_T
from ..utils.chuliu_edmonds_utils import chuliu_edmonds_one_root
from ..utils.scores_and_losses_utils import (
    compute_acc_class,
    compute_acc_deprel,
    compute_acc_head,
    compute_LAS,
    compute_LAS_chuliu,
    compute_loss_class,
    compute_loss_deprel,
    compute_loss_head,
)
from ..utils.types import (
    CONFIG_FILE_NAME,
    MODEL_FILE_NAME,
    DataclassJSONEncoder,
    ModelParams_T,
    TrainingConfig,
)
from ..utils.ud_dataset import (
    SequencePredictionBatch_T,
    SequenceTrainingBatch_T,
    UDDataset,
)
from .BertForDepRelOutput import BertForDeprelBatchOutput
from .PosAndDepRelParserHead import PosAndDeprelParserHead


@dataclass
class EvalResultAccumulator:
    good_head_epoch, total_head_epoch, loss_head_epoch = 0.0, 0.0, 0.0
    good_uposs_epoch, total_uposs_epoch, total_loss_uposs_epoch = 0.0, 0.0, 0.0
    good_xposs_epoch, total_xposs_epoch, total_loss_xposs_epoch = 0.0, 0.0, 0.0
    good_feats_epoch, total_feats_epoch, total_loss_feats_epoch = 0.0, 0.0, 0.0
    (
        good_lemma_scripts_epoch,
        total_lemma_scripts_epoch,
        total_loss_lemma_scripts_epoch,
    ) = (
        0.0,
        0.0,
        0.0,
    )
    good_deprel_epoch, total_deprel_epoch, loss_deprel_epoch = 0.0, 0.0, 0.0
    n_correct_LAS_epoch, n_correct_LAS_epoch, total_LAS_epoch = 0.0, 0.0, 0.0
    n_correct_LAS_chuliu_epoch, total_LAS_epoch = 0.0, 0.0
    total_sents = 0
    criterion: CrossEntropyLoss

    def accumulate(
        self,
        batch: SequenceTrainingBatch_T,
        model_output: BertForDeprelBatchOutput,
        chuliu_heads_pred: torch.Tensor,
    ):
        self.total_sents += len(batch.idx)

        n_correct_LAS_batch, total_LAS_batch = compute_LAS(
            model_output.heads, model_output.deprels, batch.heads, batch.deprels
        )
        n_correct_LAS_chuliu_batch, total_LAS_batch = compute_LAS_chuliu(
            chuliu_heads_pred, model_output.deprels, batch.heads, batch.deprels
        )
        self.n_correct_LAS_epoch += n_correct_LAS_batch
        self.n_correct_LAS_chuliu_epoch += n_correct_LAS_chuliu_batch
        self.total_LAS_epoch += total_LAS_batch

        loss_head_batch = compute_loss_head(
            model_output.heads, batch.heads, self.criterion
        )
        good_head_batch, total_head_batch = compute_acc_head(
            model_output.heads, batch.heads
        )
        self.loss_head_epoch += loss_head_batch.item()
        self.good_head_epoch += good_head_batch
        self.total_head_epoch += total_head_batch

        loss_deprel_batch = compute_loss_deprel(
            model_output.deprels, batch.deprels, batch.heads, self.criterion
        )
        good_deprel_batch, total_deprel_batch = compute_acc_deprel(
            model_output.deprels, batch.deprels, batch.heads
        )
        self.loss_deprel_epoch += loss_deprel_batch.item()
        self.good_deprel_epoch += good_deprel_batch
        self.total_deprel_epoch += total_deprel_batch

        good_uposs_batch, total_uposs_batch = compute_acc_class(
            model_output.uposs, batch.uposs
        )
        self.good_uposs_epoch += good_uposs_batch
        self.total_uposs_epoch += total_uposs_batch

        good_xposs_batch, total_xposs_batch = compute_acc_class(
            model_output.xposs, batch.xposs
        )
        self.good_xposs_epoch += good_xposs_batch
        self.total_xposs_epoch += total_xposs_batch

        good_feats_batch, total_feats_batch = compute_acc_class(
            model_output.feats, batch.feats
        )
        self.good_feats_epoch += good_feats_batch
        self.total_feats_epoch += total_feats_batch

        good_lemma_scripts_batch, total_lemma_scripts_batch = compute_acc_class(
            model_output.lemma_scripts, batch.lemma_scripts
        )
        self.good_lemma_scripts_epoch += good_lemma_scripts_batch
        self.total_lemma_scripts_epoch += total_lemma_scripts_batch

        loss_uposs_batch = compute_loss_class(
            model_output.uposs, batch.uposs, self.criterion
        )
        self.total_loss_uposs_epoch += loss_uposs_batch.item()

        loss_xposs_batch = compute_loss_class(
            model_output.xposs, batch.xposs, self.criterion
        )
        self.total_loss_xposs_epoch += loss_xposs_batch.item()

        loss_feats_batch = compute_loss_class(
            model_output.feats, batch.feats, self.criterion
        )
        self.total_loss_feats_epoch += loss_feats_batch.item()

        loss_lemma_scripts_batch = compute_loss_class(
            model_output.lemma_scripts, batch.lemma_scripts, self.criterion
        )
        self.total_loss_lemma_scripts_epoch += loss_lemma_scripts_batch.item()

    def get_results(self, ndigits=3):
        loss_head_epoch = self.loss_head_epoch / self.total_sents
        acc_head_epoch = self.good_head_epoch / self.total_head_epoch

        loss_deprel_epoch = self.loss_deprel_epoch / self.total_sents
        acc_deprel_epoch = self.good_deprel_epoch / self.total_deprel_epoch

        loss_uposs_epoch = self.total_loss_uposs_epoch / self.total_sents
        acc_uposs_epoch = self.good_uposs_epoch / self.total_uposs_epoch

        loss_xposs_epoch = self.total_loss_xposs_epoch / self.total_sents
        acc_xposs_epoch = self.good_xposs_epoch / self.total_xposs_epoch

        loss_feats_epoch = self.total_loss_feats_epoch / self.total_sents
        acc_feats_epoch = self.good_feats_epoch / self.total_feats_epoch

        loss_lemma_scripts_epoch = (
            self.total_loss_lemma_scripts_epoch / self.total_sents
        )
        acc_lemma_scripts_epoch = (
            self.good_lemma_scripts_epoch / self.total_lemma_scripts_epoch
        )

        LAS_epoch = self.n_correct_LAS_epoch / self.total_LAS_epoch
        LAS_chuliu_epoch = self.n_correct_LAS_chuliu_epoch / self.total_LAS_epoch

        loss_epoch = (
            loss_head_epoch
            + loss_deprel_epoch
            + loss_uposs_epoch
            + loss_xposs_epoch
            + loss_feats_epoch
            + loss_lemma_scripts_epoch
        )
        loss_epoch /= 6

        return EvalResult(
            LAS_epoch=LAS_epoch,
            LAS_chuliu_epoch=LAS_chuliu_epoch,
            acc_head_epoch=acc_head_epoch,
            acc_deprel_epoch=acc_deprel_epoch,
            acc_uposs_epoch=acc_uposs_epoch,
            acc_xposs_epoch=acc_xposs_epoch,
            acc_feats_epoch=acc_feats_epoch,
            acc_lemma_scripts_epoch=acc_lemma_scripts_epoch,
            loss_head_epoch=loss_head_epoch,
            loss_deprel_epoch=loss_deprel_epoch,
            loss_uposs_epoch=loss_uposs_epoch,
            loss_xposs_epoch=loss_xposs_epoch,
            loss_feats_epoch=loss_feats_epoch,
            loss_lemma_scripts_epoch=loss_lemma_scripts_epoch,
            loss_epoch=loss_epoch,
        )


@dataclass
class TrainingDiagnostics:
    n_sentences_train: int
    n_sentences_test: int
    epoch: int
    saved: bool
    is_best_loss: bool
    is_best_LAS: bool
    epochs_without_improvement: int
    stopping_early: bool


@dataclass
class EvalResult:
    LAS_epoch: float
    LAS_chuliu_epoch: float
    acc_head_epoch: float
    acc_deprel_epoch: float
    acc_uposs_epoch: float
    acc_xposs_epoch: float
    acc_feats_epoch: float
    acc_lemma_scripts_epoch: float
    loss_head_epoch: float
    loss_deprel_epoch: float
    loss_uposs_epoch: float
    loss_xposs_epoch: float
    loss_feats_epoch: float
    loss_lemma_scripts_epoch: float
    loss_epoch: float
    training_diagnostics: Optional[TrainingDiagnostics] = None

    def _set_diagnostic_info(self, diagnostics: TrainingDiagnostics):
        self.training_diagnostics = diagnostics

    def rounded(self, ndigits):
        return type(self)(
            **{
                k: round(v, ndigits)
                for k, v in self.__dict__.items()
                if isinstance(v, float)
            },
            training_diagnostics=self.training_diagnostics,
        )

    def __str__(self):
        return (
            "\nEpoch evaluation results\n"
            f"Average total loss = {self.loss_epoch:.3f}\n"
            f"LAS = {self.LAS_epoch:.3f}\n"
            f"LAS_chuliu = {self.LAS_chuliu_epoch:.3f}\n"
            f"Acc. head = {self.acc_head_epoch:.3f}\n"
            f"Acc. deprel = {self.acc_deprel_epoch:.3f}\n"
            f"Acc. upos = {self.acc_uposs_epoch:.3f}\n"
            f"Acc. feat = {self.acc_feats_epoch:.3f}\n"
            f"Acc. lemma_script = {self.acc_lemma_scripts_epoch:.3f}\n"
            f"Acc. xposs = {self.acc_xposs_epoch:.3f}\n"
        )


DEFAULT_MODEL_NAME = "default"


class BertForDeprel(Module):
    @staticmethod
    def load_single_pretrained_for_prediction(
        pretrained_model_path: Path, device: torch.device
    ) -> "BertForDeprel":
        """Load a pre-trained model ready to perform predictions."""
        model_params = ModelParams_T.from_model_path(pretrained_model_path)
        model = BertForDeprel(
            model_params.embedding_type,
            model_params.annotation_schema,
            device,
            {DEFAULT_MODEL_NAME: pretrained_model_path},
            DEFAULT_MODEL_NAME,
            no_classifier_heads=False,
        )
        model.eval()
        return model

    @staticmethod
    def load_pretrained_for_prediction(
        pretrained_model_paths: Dict[str, Path], active_model: str, device: torch.device
    ) -> "BertForDeprel":
        """Load a set of pre-trained models ready to perform predictions.
        pretrained_model_paths: a mapping from model names to model paths
        active_model: the name of the model to use for predictions
        device: the device to load the model on"""
        if active_model not in pretrained_model_paths:
            raise ValueError(
                f"Model {active_model} not found in {pretrained_model_paths.keys()}"
            )

        model_params = ModelParams_T.from_model_path(
            pretrained_model_paths[active_model]
        )
        for other_name, other_path in pretrained_model_paths.items():
            other_params = ModelParams_T.from_model_path(other_path)
            if other_params.embedding_type != model_params.embedding_type:
                raise ValueError(
                    "All loaded models must have the same embedding types."
                    "{active_model} has {model_params.embedding_type}, but "
                    f"{other_name} has {other_params.embedding_type}."
                )

        model = BertForDeprel(
            model_params.embedding_type,
            model_params.annotation_schema,
            device,
            pretrained_model_paths,
            active_model,
            no_classifier_heads=False,
        )
        model.eval()
        return model

    @staticmethod
    def load_pretrained_for_retraining(
        pretrained_model_path: Path,
        new_annotation_schema: AnnotationSchema_T,
        device: torch.device,
    ) -> "BertForDeprel":
        """Load a pre-trained model, but remove the prediction heads and replace the
        annotation schema."""
        model_params = ModelParams_T.from_model_path(pretrained_model_path)
        model = BertForDeprel(
            model_params.embedding_type,
            new_annotation_schema,
            device,
            {DEFAULT_MODEL_NAME: pretrained_model_path},
            DEFAULT_MODEL_NAME,
            no_classifier_heads=True,
        )
        model.train()
        return model

    @staticmethod
    def load_pretrained_for_finetuning(
        pretrained_model_path: Path,
        new_annotation_schema: AnnotationSchema_T,
        device: torch.device,
    ) -> "BertForDeprel":
        """Load a pre-trained model, incorporating the values from a new annotation
        schema into the existing one."""
        model_params = ModelParams_T.from_model_path(pretrained_model_path)
        model_params.annotation_schema.update(new_annotation_schema)
        model = BertForDeprel(
            model_params.embedding_type,
            model_params.annotation_schema,
            device,
            {DEFAULT_MODEL_NAME: pretrained_model_path},
            DEFAULT_MODEL_NAME,
            no_classifier_heads=False,
        )
        model.train()
        return model

    @staticmethod
    def new_model(
        embedding_type: str,
        annotation_schema: AnnotationSchema_T,
        device: torch.device,
    ) -> "BertForDeprel":
        """Create a new model with the given embedding type and annotation schema.
        The model will be a blank slate that must be trained."""
        model = BertForDeprel(
            embedding_type,
            annotation_schema,
            device,
            pretrained_model_paths={},
            active_model=None,
            no_classifier_heads=False,
        )
        model.train()
        return model

    def __init__(
        self,
        embedding_type: str,
        annotation_schema: AnnotationSchema_T,
        device: torch.device,
        pretrained_model_paths: Dict[str, Path],
        active_model: Optional[str],
        no_classifier_heads: bool,
    ):
        """Clients should not call this directly. Instead, use one of the static
        constructors to create a new model or load a pre-trained one."""
        super().__init__()
        self.embedding_type = embedding_type
        self.annotation_schema = annotation_schema
        self.pretrained_model_path = pretrained_model_paths
        self.device = device
        self.user_diagnostic_info = {}

        self.__init_language_model_layer(embedding_type)
        llm_hidden_size = (
            self.llm_layer.config.hidden_size
        )  # expected to get embedding size of bert custom model

        n_uposs = len(annotation_schema.uposs)
        n_xposs = len(annotation_schema.xposs)
        n_deprels = len(annotation_schema.deprels)
        n_feats = len(annotation_schema.feats)
        n_lemma_scripts = len(annotation_schema.lemma_scripts)
        self.tagger_layer = PosAndDeprelParserHead(
            n_uposs, n_deprels, n_feats, n_lemma_scripts, n_xposs, llm_hidden_size
        )

        if pretrained_model_paths:
            print("Loading weights of the pretrained model(s)")
            self.__load_pretrained_checkpoints(
                pretrained_model_paths,
            )
            self._active_model = active_model
            self.__apply_pretrained_checkpoint(
                self._checkpoints[self._active_model], no_classifier_heads
            )

        self.total_trainable_parameters = self.get_total_trainable_parameters()
        print("TOTAL TRAINABLE PARAMETERS : ", self.total_trainable_parameters)

        self._set_criterions_and_optimizer()

        self.to(device)

    def __init_language_model_layer(self, embedding_type):
        # TODO: user gets to choose the type here, so it's wrong to
        # assume XLMRobertaModel
        self.llm_layer: XLMRobertaModel = AutoModel.from_pretrained(embedding_type)
        self.llm_layer.config
        adapter_config = PfeifferConfig(reduction_factor=4, non_linearity="gelu")
        adapter_name = "Pfeiffer_gelu"
        self.llm_layer.add_adapter(adapter_name, config=adapter_config)
        # TODO: this should only be set when the mode is "train";
        self.llm_layer.train_adapter([adapter_name])
        self.llm_layer.set_active_adapters([adapter_name])

    def _set_criterions_and_optimizer(self):
        self.criterion = CrossEntropyLoss(ignore_index=-1)
        self.optimizer = AdamW(self.parameters(), lr=0.00005)
        print("Criterion and Optimizer set")

    @property
    def max_position_embeddings(self):
        return self.llm_layer.config.max_position_embeddings

    def get_total_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def add_diagnostic(self, key: str, value: Any):
        """Add any (JSON-writeable) diagnostic key/value that will help with
        understanding the model. This is written into the model config file when the
        model is saved."""
        self.user_diagnostic_info[key] = value

    def to(self, device: torch.device) -> Self:
        new_model = super().to(device)
        new_model.device = device
        return new_model

    def encode_dataset(self, sentences: Iterable[sentenceJson_T]) -> UDDataset:
        """Convert sentences into a dataset encoded for use with the model
        (training or prediction). Note that the returned dataset may not contain all
        of the input sentences, if some of them are too long or otherwise invalid.
        The client will need to check the IDs of the returned sentences to see which
        ones were included/excluded."""
        return UDDataset(
            iter(sentences),
            self.annotation_schema,
            self.embedding_type,
            self.max_position_embeddings,
        )

    def forward(self, batch: SequencePredictionBatch_T) -> BertForDeprelBatchOutput:
        batch = batch.to(self.device)

        # Feed the input to BERT model to obtain contextualized representations
        bert_output = self.llm_layer.forward(
            batch.sequence_token_ids, attention_mask=batch.attn_masks, return_dict=True
        )
        # return_dict is True, so the return value will never be a tuple, but we do
        # this to satisfy the type-checker
        assert not isinstance(bert_output, tuple)

        x = bert_output.last_hidden_state
        output = self.tagger_layer.forward(x, batch)
        return output

    def __compute_loss(
        self, batch: SequenceTrainingBatch_T, preds: BertForDeprelBatchOutput
    ):
        loss_batch = compute_loss_head(preds.heads, batch.heads, self.criterion)
        loss_batch += compute_loss_deprel(
            preds.deprels, batch.deprels, batch.heads.clone(), self.criterion
        )
        loss_batch += compute_loss_class(preds.uposs, batch.uposs, self.criterion)
        loss_batch += compute_loss_class(preds.xposs, batch.xposs, self.criterion)
        loss_batch += compute_loss_class(preds.feats, batch.feats, self.criterion)
        loss_batch += compute_loss_class(
            preds.lemma_scripts, batch.lemma_scripts, self.criterion
        )
        return loss_batch

    def train_epoch(self, loader):
        time_from_start = 0
        parsing_speed = 0
        start = timer()
        self.train()
        processed_sentence_counter = 0
        total_number_batch = len(loader)
        print_every = max(
            1, total_number_batch // 8
        )  # so we print only around 8 times per epochs
        batch: SequenceTrainingBatch_T
        for batch_counter, batch in enumerate(loader):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            preds = self.forward(batch)
            loss_batch = self.__compute_loss(batch, preds)
            loss_batch.backward()
            self.optimizer.step()

            processed_sentence_counter += batch.sequence_token_ids.size(0)
            time_from_start = timer() - start
            parsing_speed = int(
                round(((processed_sentence_counter + 1) / time_from_start) / 100, 2)
                * 100
            )

            if batch_counter % print_every == 0:
                print(
                    f"Training: {100 * (batch_counter + 1) / len(loader):.2f}% "
                    f"complete. {time_from_start:.2f} seconds in epoch "
                    f"({parsing_speed:.2f} sents/sec)",
                    flush=True,
                )
        # My Mac runs out of shared memory without this. See
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # TODO: do this actually help?
        if self.device == "mps":
            torch.mps.empty_cache()
        print(
            f"Finished training epoch in {time_from_start:.2f} seconds ("
            f"{processed_sentence_counter} sentence at {parsing_speed} sents/sec)"
        )

    def eval_on_dataset(self, loader) -> EvalResult:
        """Evaluate the model's performance on the given gold-annotated data."""
        self.eval()
        with torch.no_grad():
            num_batches = len(loader)
            print_every = max(1, num_batches // 4)
            results_accumulator = EvalResultAccumulator(self.criterion)
            start = timer()
            processed_sentence_counter = 0

            batch: SequenceTrainingBatch_T
            for batch_counter, batch in enumerate(loader):
                batch = batch.to(self.device)

                model_output = self.forward(batch).detach()
                chuliu_heads_pred = self.chuliu_heads_pred(batch, model_output)
                results_accumulator.accumulate(batch, model_output, chuliu_heads_pred)

                processed_sentence_counter += batch.sequence_token_ids.size(0)
                time_from_start = timer() - start
                parsing_speed = int(
                    round(((processed_sentence_counter + 1) / time_from_start) / 100, 2)
                    * 100
                )

                if batch_counter % print_every == 0:
                    print(
                        f"Evaluating: {100 * (batch_counter + 1) / len(loader):.2f}% "
                        f"complete. {time_from_start:.2f} seconds in epoch ("
                        f"{parsing_speed:.2f} sents/sec)",
                        flush=True,
                    )

            results = results_accumulator.get_results()

        return results

    # TODO: combine with Predictor.__get_constrained_dependencies
    def chuliu_heads_pred(
        self, batch: SequenceTrainingBatch_T, model_output: BertForDeprelBatchOutput
    ) -> torch.Tensor:
        chuliu_heads_pred = batch.heads.clone()
        for i_sentence, (
            heads_pred_sentence,
            tok_starts_word_sentence,
            idx_converter_sentence,
        ) in enumerate(
            zip(model_output.heads, batch.tok_starts_word, batch.idx_converter)
        ):
            # clone and set the value for the leading CLS token to True so that
            # Chu-Liu/Edmonds has the dummy root node it requires.
            tok_starts_word_or_is_root = tok_starts_word_sentence.clone()
            tok_starts_word_or_is_root[0] = True
            # Get the head scores for each word predicted
            heads_pred_np = heads_pred_sentence[:, tok_starts_word_or_is_root][
                tok_starts_word_or_is_root
            ]
            # Chu-Liu/Edmonds implementation requires numpy array, which can only be
            # created in CPU memory
            heads_pred_np = heads_pred_np.cpu().numpy()

            # TODO: why transpose? C-L/E wants (dep, head), which is what we have.
            # Unless the constraint logic for prediction is wrong, which is possible...
            chuliu_heads_vector = chuliu_edmonds_one_root(
                np.transpose(heads_pred_np, (1, 0))
            )
            # Remove the dummy root node for final output
            chuliu_heads_vector = chuliu_heads_vector[1:]

            for i_dependent_word, chuliu_head_pred in enumerate(chuliu_heads_vector):
                chuliu_heads_pred[
                    i_sentence, idx_converter_sentence[i_dependent_word + 1]
                ] = idx_converter_sentence[chuliu_head_pred]
        # TODO: what is this return value?
        return chuliu_heads_pred

    def save_model(self, model_dir: Path, training_config: TrainingConfig):
        trainable_weight_names = [
            n for n, p in self.llm_layer.named_parameters() if p.requires_grad
        ] + [n for n, p in self.tagger_layer.named_parameters() if p.requires_grad]
        state = {"adapter": {}, "tagger": {}}
        for k, v in self.llm_layer.state_dict().items():
            if k in trainable_weight_names:
                state["adapter"][k] = v
        for k, v in self.tagger_layer.state_dict().items():
            if k in trainable_weight_names:
                state["tagger"][k] = v

        ckpt_fpath: Path = model_dir / MODEL_FILE_NAME
        config_path: Path = model_dir / CONFIG_FILE_NAME
        torch.save(state, ckpt_fpath)
        print(
            "Saving adapter weights to ... {} ({:.2f} MB)".format(
                ckpt_fpath, ckpt_fpath.stat().st_size * 1.0 / (1024 * 1024)
            )
        )
        with open(config_path, "w") as outfile:
            config = {
                "annotation_schema": self.annotation_schema,
                "embedding_type": self.embedding_type,
                "max_epoch": training_config.max_epochs,
                "patience": training_config.patience,
                "batch_size": training_config.batch_size,
                "num_workers": training_config.num_workers,
                "saved_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "diagnostics": self.user_diagnostic_info,
            }
            outfile.write(
                json.dumps(
                    config,
                    ensure_ascii=False,
                    indent=4,
                    cls=DataclassJSONEncoder,
                )
            )

    def activate(self, model_name: str):
        """Activate an already-loaded pretrained model.
        model_name: which loaded model to activate.
        """
        if model_name not in self._checkpoints:
            raise ValueError(
                f"Specified model name {model_name} not found among loaded models: "
                f"{self._checkpoints.keys()}"
            )
        self.__apply_pretrained_checkpoint(self._checkpoints[self._active_model])
        self._active_model = model_name

    def __load_pretrained_checkpoints(self, model_paths: Dict[str, Path]):
        self._checkpoints = {}
        for model_name, model_path in model_paths.items():
            checkpoint_path = model_path / MODEL_FILE_NAME
            checkpoint = torch.load(checkpoint_path)
            self._checkpoints[model_name] = checkpoint

    def __apply_pretrained_checkpoint(
        self, checkpoint_state, no_classifier_heads=False
    ):
        tagger_pretrained_dict = self.tagger_layer.state_dict()
        for layer_name, weights in checkpoint_state["tagger"].items():
            if no_classifier_heads and layer_name in [
                "deprel.pairwise_weight",
                "uposs_ffn.weight",
                "uposs_ffn.bias",
                "xposs_ffn.weight",
                "xposs_ffn.bias",
                "lemma_scripts_ffn.weight",
                "lemma_scripts_ffn.bias",
                "feats_ffn.weight",
                "feats_ffn.bias",
            ]:
                print(f"Overwriting pretrained layer {layer_name}")
                continue
            tagger_pretrained_dict[layer_name] = weights
        self.tagger_layer.load_state_dict(tagger_pretrained_dict)

        llm_pretrained_dict = self.llm_layer.state_dict()
        for layer_name, weights in checkpoint_state["adapter"].items():
            if layer_name in llm_pretrained_dict:
                llm_pretrained_dict[layer_name] = weights
        self.llm_layer.load_state_dict(llm_pretrained_dict)
        return


# To reactivate if problem in the loading of the model states
# loaded_state_dict = OrderedDict()
# for k, v in checkpoint["state_dict"].items():
#     name = k.replace("module.", "")
#     loaded_state_dict[name] = v
