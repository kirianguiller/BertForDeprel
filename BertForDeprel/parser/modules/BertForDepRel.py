import json
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional

import numpy as np
import torch
import torch.mps
from torch.nn import CrossEntropyLoss, Module
from torch.optim import AdamW
from transformers import (  # type: ignore (TODO: why can't PyLance find these?)
    AutoModel,
    XLMRobertaModel,
)
from transformers.adapters import PfeifferConfig

from ..utils.annotation_schema import AnnotationSchema_T
from ..utils.chuliu_edmonds_utils import chuliu_edmonds_one_root
from ..utils.load_data_utils import SequencePredictionBatch_T, SequenceTrainingBatch_T
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
from ..utils.types import DataclassJSONEncoder, ModelParams_T, TrainingConfig
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
            f"Total loss = {self.loss_epoch:.3f}\n"
            f"LAS = {self.LAS_epoch:.3f}\n"
            f"LAS_chuliu = {self.LAS_chuliu_epoch:.3f}\n"
            f"Acc. head = {self.acc_head_epoch:.3f}\n"
            f"Acc. deprel = {self.acc_deprel_epoch:.3f}\n"
            f"Acc. upos = {self.acc_uposs_epoch:.3f}\n"
            f"Acc. feat = {self.acc_feats_epoch:.3f}\n"
            f"Acc. lemma_script = {self.acc_lemma_scripts_epoch:.3f}\n"
            f"Acc. xposs = {self.acc_xposs_epoch:.3f}\n"
        )


class BertForDeprel(Module):
    # TODO: for fine-tuning, we need to be sure that:
    # - the annotation schema of new model is the same as pre-trained
    # - the new model has same architecture as old one

    # TODO: separate constructor for pretrained and new would be nice
    # TODO: it's weird to have the annotation schema conceptually separated from the
    # model. Move it from config to its own file instead.

    # goals: specify embedding type and annotation schema explicitly to constructor

    @staticmethod
    def load_pretrained_for_prediction(pretrained_model_path: Path) -> "BertForDeprel":
        """Load a pre-trained model ready to perform predictions."""
        model_params = ModelParams_T.from_model_path(pretrained_model_path)
        model = BertForDeprel(
            model_params.embedding_type,
            model_params.annotation_schema,
            pretrained_model_path,
            no_classifier_heads=False,
        )
        model.eval()
        return model

    @staticmethod
    def load_pretrained_for_retraining(
        pretrained_model_path: Path, new_annotation_schema: AnnotationSchema_T
    ) -> "BertForDeprel":
        """Load a pre-trained model, but remove the prediction heads and replace the
        annotation schema."""
        model_params = ModelParams_T.from_model_path(pretrained_model_path)
        model = BertForDeprel(
            model_params.embedding_type,
            new_annotation_schema,
            pretrained_model_path,
            no_classifier_heads=True,
        )
        model.train()
        return model

    @staticmethod
    def load_pretrained_for_finetuning(
        pretrained_model_path: Path, new_annotation_schema: AnnotationSchema_T
    ) -> "BertForDeprel":
        """Load a pre-trained model, incorporating the values from a new annotation
        schema into the existing one."""
        model_params = ModelParams_T.from_model_path(pretrained_model_path)
        model_params.annotation_schema.update(new_annotation_schema)
        model = BertForDeprel(
            model_params.embedding_type,
            model_params.annotation_schema,
            pretrained_model_path,
            no_classifier_heads=False,
        )
        model.train()
        return model

    @staticmethod
    def new_model(
        embedding_type: str, annotation_schema: AnnotationSchema_T
    ) -> "BertForDeprel":
        """Create a new model with the given embedding type and annotation schema.
        The model will be a blank slate that must be trained."""
        model = BertForDeprel(
            embedding_type,
            annotation_schema,
            pretrained_model_path=None,
            no_classifier_heads=False,
        )
        model.train()
        return model

    def __init__(
        self,
        embedding_type: str,
        annotation_schema: AnnotationSchema_T,
        pretrained_model_path: Optional[Path] = None,
        no_classifier_heads: bool = False,
    ):
        super().__init__()
        self.embedding_type = embedding_type
        self.annotation_schema = annotation_schema
        self.pretrained_model_path = pretrained_model_path

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

        if pretrained_model_path:
            print("Loading weights of the pretrained model")
            self.__load_pretrained_adapter(pretrained_model_path, no_classifier_heads)

        self.total_trainable_parameters = self.get_total_trainable_parameters()
        print("TOTAL TRAINABLE PARAMETERS : ", self.total_trainable_parameters)

        self._set_criterions_and_optimizer()

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

    def forward(self, batch: SequencePredictionBatch_T) -> BertForDeprelBatchOutput:
        """
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used
            to avoid contribution of PAD tokens
            -mode: if set to "predict", resulting tensors will be detached before
            returning
        """

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

    def train_epoch(self, loader, device):
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
            batch = batch.to(device)
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
                    f"({parsing_speed:.2f} sents/sec)"
                )
        # My Mac runs out of shared memory without this. See
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # TODO: do this actually help?
        if device == "mps":
            torch.mps.empty_cache()
        print(
            f"Finished training epoch in {time_from_start:.2f} seconds ("
            f"{processed_sentence_counter} sentence at {parsing_speed} sents/sec)"
        )

    def eval_on_dataset(self, loader, device) -> EvalResult:
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
                batch = batch.to(device, is_eval=True)

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
                        f"{parsing_speed:.2f} sents/sec)"
                    )

            results = results_accumulator.get_results()
            print(results)

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

        ckpt_fpath: Path = model_dir / "model.pt"
        config_path: Path = model_dir / "config.json"
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
            }
            outfile.write(
                json.dumps(
                    config,
                    ensure_ascii=False,
                    indent=4,
                    cls=DataclassJSONEncoder,
                )
            )

    def __load_pretrained_adapter(self, model_path: Path, no_classifier_heads=False):
        ckpt_fpath = model_path / "model.pt"
        checkpoint_state = torch.load(ckpt_fpath)

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
