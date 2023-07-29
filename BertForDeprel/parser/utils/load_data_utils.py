from dataclasses import dataclass
import os
from typing import Dict, List, Any, Self, Tuple, TypeVar, Literal

from conllup.conllup import sentenceJson_T, _featuresConllToJson, _featuresJsonToConll, readConlluFile

from torch.utils.data import Dataset
from torch import tensor, Tensor
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .types import ModelParams_T
from .annotation_schema_utils import compute_annotation_schema, resolve_conllu_paths, is_annotation_schema_empty, NONE_VOCAB
from .lemma_script_utils import apply_lemma_rule, gen_lemma_script


DUMMY_ID = -1

# Words and tokens are not the same! A word can be composed of multiple subword tokens.
# In the classes below, we work with (smallest to largest) tokens and words contained
# in sequences (or sentences) and batches of sequences. We'll use T for the number of
# tokens in a sequence (including added special tokens), W for the number of words in
# a sequence, and B for the number of sequences in a batch.
@dataclass
class SequencePrediction_T:
    """Index in the dataset"""
    idx: int
    """the ConllU data for the sentence"""
    sentence_json: sentenceJson_T

    """The token ids of the sequence, prepended with a CLS token and appended with a SEP token as required
    by BERT models. Size is (T)."""
    sequence_token_ids: List[int]
    """1 if sequence token begins a new word, 0 otherwise. Size is (T)."""
    subwords_start: List[int]

    """Maps word index + 1 to the index in the sequence_token_ids where the word begins. Size is (W)."""
    idx_converter: List[int]
    """The number of tokens in each input word. Size is (W)."""
    tokens_len: List[int]

Sequence_T = TypeVar('Sequence_T', bound=SequencePrediction_T)

class SequenceTraining_T(SequencePrediction_T):
    """Each list is size (T) and gives the ID of the class label for the corresponding
    token in the sequence, where tokens that do not begin a new word are given a dummy
    class value."""
    uposs: List[int]
    xposs: List[int]
    heads: List[int]
    deprels: List[int]
    feats: List[int]
    lemma_scripts: List[int]

    def __init__(self, pred_data: SequencePrediction_T, uposs: List[int], xposs: List[int], heads: List[int], deprels: List[int], feats: List[int], lemma_scripts: List[int]):
        super().__init__(**pred_data.__dict__)
        self.uposs = uposs
        self.xposs = xposs
        self.heads = heads
        self.deprels = deprels
        self.feats = feats
        self.lemma_scripts = lemma_scripts

# batch classes are for entire datasets of parses
@dataclass
class SequencePredictionBatch_T:
    """Except for attn_masks and max_sentence_legth, each field contains all of the tensors of the corresponding field in
    SequencePrediction_T for each sequence in the batch. See that class for more details.
    Sizes are then (B, T or W) for each batched field."""
    idx: Tensor
    sequence_token_ids: Tensor
    # Tensor of shape [batch_size, max_seq_length] containing attention masks to be used to avoid contribution of PAD tokens.
    # Size is (B, T). See https://huggingface.co/docs/transformers/glossary#attention-mask.
    attn_masks: Tensor

    subwords_start: Tensor
    idx_converter: Tensor

    # The maximum length of any sequence in the batch, determining the size of the tensors representing sequences
    # (shorter sequences are padded).
    max_sentence_length: int

    def to(self, device: str, is_eval=False) -> Self:
        """Returns a new training batch with the tensors sent to the specified device. For use
        during model training or prediction (is_eval=False) or evaluation (is_eval=True)."""
        if is_eval:
            subwords_start = self.subwords_start.to(device)
            idx_converter = self.idx_converter.to(device)
        else:
            subwords_start = self.subwords_start
            idx_converter = self.idx_converter
        return SequencePredictionBatch_T(
            idx=self.idx,
            sequence_token_ids=self.sequence_token_ids.to(device),
            attn_masks=self.attn_masks.to(device),
            subwords_start=subwords_start,
            idx_converter=idx_converter,
            max_sentence_length=self.max_sentence_length
        )


@dataclass
class SequenceTrainingBatch_T(SequencePredictionBatch_T):
    """Each field contains all of the tensors of the corresponding field in
    SequenceTraining_T for each sequence in the batch. See that class for more details.
    Sizes are then (B, T) for each batched field."""
    uposs: Tensor
    xposs: Tensor
    heads: Tensor
    deprels: Tensor
    feats: Tensor
    lemma_scripts: Tensor

    def __init__(self, pred_data: SequencePredictionBatch_T, uposs: Tensor, xposs: Tensor, heads: Tensor, deprels: Tensor, feats: Tensor, lemma_scripts: Tensor):
        super().__init__(**pred_data.__dict__)
        self.uposs = uposs
        self.xposs = xposs
        self.heads = heads
        self.deprels = deprels
        self.feats = feats
        self.lemma_scripts = lemma_scripts

    def to(self, device: str, is_eval=False):
        """Returns a new training batch with the tensors sent to the specified device. For use
        during model training (is_eval=False) or evaluation (is_eval=True)."""
        return SequenceTrainingBatch_T(
            pred_data=super().to(device, is_eval),
            heads=self.heads.to(device),
            deprels=self.deprels.to(device),
            uposs=self.uposs.to(device),
            xposs=self.xposs.to(device),
            feats=self.feats.to(device),
            lemma_scripts=self.lemma_scripts.to(device),
        )


CopyOption = Literal["NONE", "EXISTING", "ALL"]


@dataclass
class PartialPredictionConfig:
    keep_upos: CopyOption="NONE"
    keep_xpos: CopyOption="NONE"
    keep_heads: CopyOption="NONE"
    keep_deprels: CopyOption="NONE"
    keep_feats: CopyOption="NONE"
    keep_lemmas: CopyOption="NONE"


class ConlluDataset(Dataset):
    def __init__(self, path_file_or_folder: str, model_params: ModelParams_T, run_mode: Literal["train", "predict"], compute_annotation_schema_if_not_found = False):
        paths = resolve_conllu_paths(path_file_or_folder)
        if is_annotation_schema_empty(model_params.annotation_schema):
            if compute_annotation_schema_if_not_found == True:
                model_params.annotation_schema = compute_annotation_schema(*paths)
            else:
                raise Exception("No annotation schema found in `model_params` while `compute_annotation_schema_if_not_found` is set to False")

        self.model_params = model_params
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer: (PreTrainedTokenizer | PreTrainedTokenizerFast) = AutoTokenizer.from_pretrained(model_params.embedding_type)

        self.run_mode = run_mode

        if self.tokenizer.cls_token_id == None:
            raise Exception("CLS token not found in tokenizer")
        self.CLS_token_id = self.tokenizer.cls_token_id

        if self.tokenizer.sep_token_id == None:
            raise Exception("SEP token not found in tokenizer")
        self.SEP_token_id = self.tokenizer.sep_token_id

        if self.tokenizer.unk_token_id == None:
            raise Exception("UNK token not found in tokenizer")
        self.UNK_token_id = self.tokenizer.unk_token_id

        if self.tokenizer.pad_token_id == None:
            raise Exception("PAD token not found in tokenizer")
        self.PAD_token_id = self.tokenizer.pad_token_id

        # 0 (CLS), 2 (SEP), 3 (UNK), 1 (PAD)
        print(f"Special tokens are {self.CLS_token_id} (CLS), {self.SEP_token_id} (SEP), {self.UNK_token_id} (UNK), {self.tokenizer.pad_token_id} (PAD)")

        self.dep2i, _ = self._compute_labels2i(self.model_params.annotation_schema.deprels)
        self.upos2i, _ = self._compute_labels2i(self.model_params.annotation_schema.uposs)
        self.xpos2i, _ = self._compute_labels2i(self.model_params.annotation_schema.xposs)
        self.feat2i, _ = self._compute_labels2i(self.model_params.annotation_schema.feats)
        self.lem2i, _ = self._compute_labels2i(self.model_params.annotation_schema.lemma_scripts)

        self._load_conll(*paths)


    def _load_conll(self, *paths):
        sentences_json: List[sentenceJson_T] = []
        for path in paths:
            sentences_json += readConlluFile(path, keep_empty_trees=False)

        self.sequences: List[SequencePrediction_T] = []
        valid_sentence_counter = 0
        for sentence_json in sentences_json:
            sequence = self._get_processed(sentence_json, valid_sentence_counter)
            # We save one spot each for CLS_token_id and SEP_token_id
            if len(sequence.sequence_token_ids) > self.model_params.max_position_embeddings - 2:
                print("Discarding sentence", len(sequence.sequence_token_ids))
                continue
            self.sequences.append(sequence)
            valid_sentence_counter += 1


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def _compute_labels2i(self, list_labels):
        sorted_set_labels = sorted(set(list_labels))

        labels2i = {}
        i2labels = {}

        for i, labels in enumerate(sorted_set_labels):
            labels2i[labels] = i
            i2labels[i] = labels

        return labels2i, i2labels

    def _pad_list(self, l: List[Any], padding_value: int, maxlen: int):
        if len(l) > maxlen:
            print(l, len(l))
            raise Exception(f"The sequence length (={len(l)}) than the maximum allowed length ({maxlen})")

        return l + [padding_value] * (maxlen - len(l))


    def _get_input(self, sequence: sentenceJson_T, idx: int) -> SequencePrediction_T:
        sequence_ids = [self.CLS_token_id]
        subwords_start = [DUMMY_ID]

        idx_converter = [0]
        tokens_len = [1]

        for input_token in sequence["treeJson"]["nodesJson"].values():
            if type(input_token["ID"]) != str:
                continue
            form = input_token["FORM"]
            # Note that we are simply ignoring un-recognized parts of the form; if all parts are unrecognized,
            # then we replace with UNK. TODO: we need add_special_tokens to be False, but we still need UNK tokens
            token_ids = self.tokenizer.encode(form, add_special_tokens=False)
            if len(token_ids) == 0:
                print(f"WARNING: Input token {input_token['ID']} ('{form}') of sentence {sequence['metaJson']['sent_id']} is not present in the tokenizer vocabulary; using UNK instead.")
                token_ids = [self.UNK_token_id]
            idx_converter.append(len(sequence_ids))
            tokens_len.append(len(token_ids))

            subword_start = [1] + [0] * (len(token_ids) - 1)
            sequence_ids += token_ids
            subwords_start += subword_start

        sequence_ids = sequence_ids + [self.SEP_token_id]
        return SequencePrediction_T(
            idx=idx,
            sentence_json=sequence,
            sequence_token_ids=sequence_ids,
            subwords_start=subwords_start,
            idx_converter=idx_converter,
            tokens_len=tokens_len,
        )

    def _get_output(self, sequence: sentenceJson_T, input: SequencePrediction_T) -> SequenceTraining_T:
        # initialize with dummy values for the leading CLS token
        uposs = [DUMMY_ID]
        xposs = [DUMMY_ID]
        heads = [DUMMY_ID]
        feats = [DUMMY_ID]
        lemma_scripts = [DUMMY_ID]
        deprels = [DUMMY_ID]
        skipped_tokens = 0

        for n_token, token in enumerate(sequence["treeJson"]["nodesJson"].values()):
            if type(token["ID"]) != str:
                skipped_tokens += 1
                continue

            token_len = input.tokens_len[n_token + 1 - skipped_tokens]
            token_padding = [DUMMY_ID] * (token_len - 1)

            upos = [get_index(token["UPOS"], self.upos2i)] + token_padding
            xpos = [get_index(token["XPOS"], self.xpos2i)] + token_padding
            feat = [get_index(_featuresJsonToConll(token["FEATS"]), self.feat2i)] + token_padding
            lemma_script = [get_index(gen_lemma_script(token["FORM"], token["LEMMA"]), self.lem2i)] + token_padding

            # TODO: how does this work for the sentence root? Is that a -1 or a self-reference?
            # Becomes 0 for the root token
            head = [sum(input.tokens_len[: token["HEAD"]])] + token_padding
            deprel = token["DEPREL"]

            deprel = [get_index(deprel, self.dep2i)] + token_padding
            # Example of what we have for a token of 2 subtokens
            # form = ["eat", "ing"]
            # pos = [4, DUMMY_ID]
            # head = [2, DUMMY_ID]
            # lemma_script = [3424, DUMMY_ID]
            # token_len = 2
            uposs += upos
            xposs += xpos
            heads += head
            deprels += deprel
            feats += feat
            lemma_scripts += lemma_script

        return SequenceTraining_T(
            pred_data=input,
            uposs=uposs,
            xposs=xposs,
            heads=heads,
            deprels=deprels,
            feats=feats,
            lemma_scripts=lemma_scripts
        )

    def _get_processed(self, sentence_json: sentenceJson_T, idx: int) -> SequencePrediction_T:
        pred_data = self._get_input(sentence_json, idx)
        if self.run_mode == "train":
            return self._get_output(sentence_json, pred_data)

        return pred_data

    def collate_fn_predict(self, sentences: List[Sequence_T]) -> SequencePredictionBatch_T:
        # Add padding values so that the entire batch has the same length, then collect the
        # field tensors for all sequences into a single tensor for each field.
        max_sentence_length  = max([len(sentence.sequence_token_ids) for sentence in sentences])
        subwords_start_batch = tensor([self._pad_list(sentence.subwords_start, DUMMY_ID, max_sentence_length) for sentence in sentences])
        idx_converter_batch  = tensor([self._pad_list(sentence.idx_converter, DUMMY_ID, max_sentence_length) for sentence in sentences])
        idx_batch            = tensor([sentence.idx for sentence in sentences])
        # The docs say to pad with the PAD token and just mask those, but for some reason we get better performance when
        # we pad with CLS and mask those (including the leading CLS).
        seq_ids_batch        = tensor([self._pad_list(sentence.sequence_token_ids, self.CLS_token_id, max_sentence_length) for sentence in sentences])
        attn_masks           = (seq_ids_batch != self.CLS_token_id).long()

        return SequencePredictionBatch_T(
            idx=idx_batch,
            sequence_token_ids=seq_ids_batch,
            subwords_start=subwords_start_batch,
            attn_masks=attn_masks,
            idx_converter=idx_converter_batch,
            max_sentence_length=max_sentence_length,
        )

    def collate_fn_train(self, sentences: List[SequenceTraining_T]) -> SequenceTrainingBatch_T:
        batch_prediction_data = self.collate_fn_predict(sentences)
        max_sentence_length = batch_prediction_data.max_sentence_length

        uposs_batch     = tensor([self._pad_list(sentence.uposs, DUMMY_ID, max_sentence_length) for sentence in sentences])
        xposs_batch     = tensor([self._pad_list(sentence.xposs, DUMMY_ID, max_sentence_length) for sentence in sentences])
        heads_batch     = tensor([self._pad_list(sentence.heads, DUMMY_ID, max_sentence_length) for sentence in sentences])
        deprels_batch   = tensor([self._pad_list(sentence.deprels, DUMMY_ID, max_sentence_length) for sentence in sentences])
        feats_batch   = tensor([self._pad_list(sentence.feats, DUMMY_ID, max_sentence_length) for sentence in sentences])
        lemma_scripts_batch   = tensor([self._pad_list(sentence.lemma_scripts, DUMMY_ID, max_sentence_length) for sentence in sentences])

        return SequenceTrainingBatch_T(
            batch_prediction_data,
            uposs=uposs_batch,
            xposs=xposs_batch,
            heads=heads_batch,
            deprels=deprels_batch,
            feats=feats_batch,
            lemma_scripts=lemma_scripts_batch,
        )

    def construct_sentence_prediction(self,
                                        idx,
                                        uposs_preds: List[int]=[],
                                        xposs_preds: List[int]=[],
                                        chuliu_heads: List[int]=[],
                                        deprels_pred_chulius: List[int]=[],
                                        feats_preds: List[int]=[],
                                        lemma_scripts_preds: List[int]=[],
                                        partial_pred_config = PartialPredictionConfig(),
                                        ) -> sentenceJson_T:
        """Constructs the final sentence structure prediction by overwriting the model's predictions with
        the input data where specified. The metadata is copied as well, since it is not predicted."""
        predicted_sentence: sentenceJson_T = self.sequences[idx].sentence_json.copy()
        tokens = list(predicted_sentence["treeJson"]["nodesJson"].values())
        annotation_schema = self.model_params.annotation_schema

        # For each of the predicted fields, we overwrite the value copied from the input with the predicted value
        # if configured to do so.
        for n_token, token in enumerate(tokens):
            if partial_pred_config.keep_upos=="NONE" or (partial_pred_config.keep_upos=="EXISTING" and token["UPOS"] == "_"):
                token["UPOS"] = annotation_schema.uposs[uposs_preds[n_token]]

            if partial_pred_config.keep_xpos == "NONE" or (partial_pred_config.keep_xpos=="EXISTING" and token["XPOS"] == "_"):
                token["XPOS"] = annotation_schema.xposs[xposs_preds[n_token]]

            if partial_pred_config.keep_heads == "NONE" or (partial_pred_config.keep_heads == "EXISTING" and token["HEAD"] == DUMMY_ID): # this one is special as for keep_heads == "EXISTING", we already handled the case earlier in the code
                token["HEAD"] = chuliu_heads[n_token]

            if partial_pred_config.keep_deprels == "NONE" or (partial_pred_config.keep_deprels=='EXISTING' and token["DEPREL"] == "_"):
                token["DEPREL"] = annotation_schema.deprels[deprels_pred_chulius[n_token]]

            if partial_pred_config.keep_feats == "NONE" or (partial_pred_config.keep_feats=="EXISTING" and token["FEATS"] == {}):
                token["FEATS"] = _featuresConllToJson(annotation_schema.feats[feats_preds[n_token]])

            if partial_pred_config.keep_lemmas == "NONE" or (partial_pred_config.keep_lemmas=="EXISTING" and token["LEMMA"] == "_"):
                lemma_script = annotation_schema.lemma_scripts[lemma_scripts_preds[n_token]]
                token["LEMMA"] = apply_lemma_rule(token["FORM"], lemma_script)
        return predicted_sentence


    def get_constrained_dependency_for_chuliu(self, idx: int) -> List[Tuple]:
        forced_relations: List[Tuple] = []

        sentence_json: sentenceJson_T = self.sequences[idx].sentence_json
        for token_json in sentence_json["treeJson"]["nodesJson"].values():
            if token_json["HEAD"] >= 0:
                forced_relations.append((int(token_json["ID"]), token_json["HEAD"]))

        return forced_relations


def get_index(label: str, mapping: Dict[str, int]) -> int:
    """
    label: a string that represent the label whose integer is required
    mapping: a dictionary with a set of labels as keys and index integers as values

    return : index (int)
    """
    index = mapping.get(label, DUMMY_ID)

    if index == DUMMY_ID:
        index = mapping[NONE_VOCAB]
        print(
            f"LOG: label '{label}' was not found in the label2index mapping. Using the index for '{NONE_VOCAB}' instead."
        )
    return index

