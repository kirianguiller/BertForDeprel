from typing import Dict, List, Any, TypedDict, Literal

from conllup.conllup import sentenceConllToJson, sentenceJson_T, _featuresConllToJson, _featuresJsonToConll

from torch.utils.data import Dataset
from torch import tensor, Tensor
from transformers import AutoTokenizer, RobertaTokenizer

from .types import ModelParams_T
from .annotation_schema_utils import compute_annotation_schema, get_path_of_conllus_from_folder_path, is_annotation_schema_empty
from .lemma_script_utils import apply_lemma_rule, gen_lemma_script

class SequenceInput_T(TypedDict):
    seq_ids: List[int]
    attn_masks: List[int]
    subwords_start: List[int]
    idx_convertor: List[int]
    tokens_len: List[int]

class SequenceOutput_T(TypedDict):
    uposs: List[int]
    heads: List[int]
    deprels: List[int]
    feats: List[int]
    lemma_scripts: List[int]

class Sequence_T(TypedDict):
    idx: List[int]
    # SequenceInput_T
    seq_ids: List[int]
    attn_masks: List[int]
    subwords_start: List[int]
    idx_convertor: List[int]
    tokens_len: List[int]
    # SequenceOutput_T
    uposs: List[int]
    heads: List[int]
    deprels: List[int]
    feats: List[int]
    lemma_scripts: List[int]


class SequenceBatch_T(TypedDict):
    idx: Tensor
    # SequenceInput_T
    seq_ids: Tensor
    attn_masks: Tensor
    subwords_start: Tensor
    idx_convertor: Tensor
    tokens_len: Tensor
    # SequenceOutput_T
    uposs: Tensor
    heads: Tensor
    deprels: Tensor
    feats: Tensor
    lemma_scripts: Tensor

class ConlluDataset(Dataset):
    def __init__(self, path_file_or_folder: str, model_params: ModelParams_T, run_mode: Literal["train", "predict"], compute_annotation_schema_if_not_found = False):
        paths = get_path_of_conllus_from_folder_path(path_file_or_folder)
        if is_annotation_schema_empty(model_params["annotation_schema"]):
            if compute_annotation_schema_if_not_found == True:
                model_params["annotation_schema"] = compute_annotation_schema(*paths)
            else:
                raise Exception("No annotation schema found in `model_params` while `compute_annotation_schema_if_not_found` is set to False")
        
        self._load_conll(*paths)

        self.model_params = model_params
        self.tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(model_params["embedding_type"])

        self.run_mode = run_mode

        self.CLS_token_id = self.tokenizer.cls_token_id
        self.SEP_token_id = self.tokenizer.sep_token_id

        self.dep2i, _ = self._compute_labels2i(self.model_params["annotation_schema"]["deprels"])
        self.upos2i, _ = self._compute_labels2i(self.model_params["annotation_schema"]["uposs"])
        self.feat2i, _ = self._compute_labels2i(self.model_params["annotation_schema"]["feats"])
        self.lem2i, _ = self._compute_labels2i(self.model_params["annotation_schema"]["lemma_scripts"])
        # self.lemma_script2i, self.i2lemma_script = self._compute_labels2i(self.args.list_lemma_script)


    def _load_conll(self, *paths):
        self.sequences = []
        for path in paths:
            print("Loading ", path)
            with open(path, "r") as infile:
                for sentence_conll in infile.read().split("\n\n"):
                    if sentence_conll.strip():
                        self.sequences.append(sentenceConllToJson(sentence_conll))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self._get_processed(idx)

    def _mount_dep2i(self, list_deprel):
        i2dr = {}
        dr2i = {}

        for idx, deprel in enumerate(list_deprel):
            i2dr[idx] = deprel
            dr2i[deprel] = idx

        return dr2i, i2dr

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
            raise Exception("The sequence is bigger than the size of the tensor")

        return l + [padding_value] * (maxlen - len(l))
    
    def _trunc(self, tensor):
        if len(tensor) >= self.model_params["maxlen"]:
            tensor = tensor[: self.model_params["maxlen"] - 1]
        return tensor

    def _get_input(self, sequence: sentenceJson_T) -> SequenceInput_T:
        sequence_ids = [self.CLS_token_id]
        subwords_start = [-1]
        idx_convertor = [0]
        tokens_len = [1]

        for token in sequence["treeJson"]["nodesJson"].values():
            if type(token["ID"]) != str:
                continue

            form = token["FORM"]
            token_ids = self.tokenizer.encode(form, add_special_tokens=False)
            idx_convertor.append(len(sequence_ids))
            tokens_len.append(len(token_ids))
            subword_start = [1] + [0] * (len(token_ids) - 1)
            sequence_ids += token_ids
            subwords_start += subword_start

        sequence_ids = self._trunc(sequence_ids)
        subwords_start = self._trunc(subwords_start)
        idx_convertor = self._trunc(idx_convertor)

        sequence_ids = sequence_ids + [self.SEP_token_id]

        sequence_ids = sequence_ids
        subwords_start = subwords_start
        idx_convertor = idx_convertor
        attn_masks = [int(token_id > 0) for token_id in sequence_ids]
        return {
            "seq_ids": sequence_ids,
            "attn_masks": attn_masks,
            "subwords_start": subwords_start,
            "idx_convertor": idx_convertor,
            "tokens_len": tokens_len,
        }

    def _get_output(self, sequence: sentenceJson_T, tokens_len) -> SequenceOutput_T:
        uposs = [-1]
        heads = [-1]
        feats = [-1]
        lemma_scripts = [-1]
        deprels = [-1]
        skipped_tokens = 0
        
        for n_token, token in enumerate(sequence["treeJson"]["nodesJson"].values()):
            if type(token["ID"]) != str:
                skipped_tokens += 1
                continue

            token_len = tokens_len[n_token + 1 - skipped_tokens]

            upos = [get_index(token["UPOS"], self.upos2i)] + [-1] * (token_len - 1)
            feat = [get_index(_featuresJsonToConll(token["FEATS"]), self.feat2i)] + [-1] * (token_len - 1)
            lemma_script = [get_index(gen_lemma_script(token["FORM"], token["LEMMA"]), self.lem2i)] + [-1] * (token_len - 1)
            
            head = [sum(tokens_len[: token["HEAD"]])] + [-1] * (token_len - 1)
            deprel = token["DEPREL"]

            deprel = [get_index(deprel, self.dep2i)] + [-1] * (
                token_len - 1
            )
            # Example of what we have for a token of 2 subtokens
            # form = ["eat", "ing"]
            # pos = [4, -1]
            # head = [2, -1]
            # lemma_script = [3424, -1]
            # token_len = 2
            uposs += upos
            heads += head
            deprels += deprel
            feats += feat
            lemma_scripts += lemma_script
        heads = self._trunc(heads)
        deprels = self._trunc(deprels)
        uposs = self._trunc(uposs)
        feats = self._trunc(feats)
        lemma_scripts = self._trunc(lemma_scripts)

        return {"uposs": uposs, "heads": heads, "deprels": deprels, "feats": feats, "lemma_scripts": lemma_scripts}

    def _get_processed(self, idx):
        processed_sequence = {"idx": idx}
        sequence = self.sequences[idx]
        sequence_input = self._get_input(sequence)
        processed_sequence.update(sequence_input)

        if self.run_mode == "train":
            sequence_output = self._get_output(
                sequence, sequence_input["tokens_len"]
            )
            processed_sequence.update(sequence_output)

        return processed_sequence

    def collate_fn(self, sentences: List[Sequence_T]) -> SequenceBatch_T:
        max_sentence_length = max([len(sentence["seq_ids"]) for sentence in sentences])
        seq_ids_batch        = tensor([self._pad_list(sentence["seq_ids"],  0, max_sentence_length) for sentence in sentences])
        subwords_start_batch = tensor([self._pad_list(sentence["subwords_start"], -1, max_sentence_length) for sentence in sentences])
        attn_masks_batch     = tensor([self._pad_list(sentence["attn_masks"],  0, max_sentence_length) for sentence in sentences])
        idx_convertor_batch  = tensor([self._pad_list(sentence["idx_convertor"], -1, max_sentence_length) for sentence in sentences])
        # tokens_len_batch  = tensor([self._pad_list(sentence["tokens_len"], -1, max_sentence_length) for sentence in sentences])
        idx_batch            = tensor([sentence["idx"] for sentence in sentences])
        collated_batch = {
            "idx": idx_batch,
            "seq_ids": seq_ids_batch,
            "subwords_start": subwords_start_batch,
            "attn_masks": attn_masks_batch,
            "idx_convertor": idx_convertor_batch,
            # "tokens_len": tokens_len_batch,
        }
        if self.run_mode == "train":
            uposs_batch     = tensor([self._pad_list(sentence["uposs"], -1, max_sentence_length) for sentence in sentences])
            heads_batch     = tensor([self._pad_list(sentence["heads"], -1, max_sentence_length) for sentence in sentences])
            deprels_batch   = tensor([self._pad_list(sentence["deprels"], -1, max_sentence_length) for sentence in sentences])
            feats_batch   = tensor([self._pad_list(sentence["feats"], -1, max_sentence_length) for sentence in sentences])
            lemma_scripts_batch   = tensor([self._pad_list(sentence["lemma_scripts"], -1, max_sentence_length) for sentence in sentences])
            collated_batch.update({"uposs": uposs_batch,
                                     "heads": heads_batch,
                                     "deprels": deprels_batch,
                                     "feats": feats_batch,
                                     "lemma_scripts": lemma_scripts_batch,
                                     })

        return collated_batch


    def add_prediction_to_sentence_json(self, idx, uposs_pred_list, chuliu_heads_list, deprels_pred_chuliu_list, feats_pred_list, lemma_scripts_pred_list, write_preds_in_misc = False):
        predicted_sentence_json: sentenceJson_T = self.sequences[idx].copy()
        tokens = list(predicted_sentence_json["treeJson"]["nodesJson"].values())
        annotation_schema = self.model_params["annotation_schema"]
        for n_token, (upos_index, head_chuliu, deprel_chuliu, feats_index, lemma_script_index) in enumerate(
                zip(
                    uposs_pred_list,
                    chuliu_heads_list,
                    deprels_pred_chuliu_list,
                    feats_pred_list,
                    lemma_scripts_pred_list,
                )
        ):
            token = tokens[n_token]
            token["MISC"] = {}
            if write_preds_in_misc:
                misc = token["MISC"]
                misc["deprel_pred"] = annotation_schema["deprels"][deprel_chuliu]

                # misc['head_MST']= str(gov_dict.get(n_token+1, 'missing_gov'))
                misc["head_MST_pred"] = str(head_chuliu)
                misc["upostag_pred"] = annotation_schema["uposs"][upos_index]
                # lemma_script = annotation_schema["i2lemma_script"][lemma_script_index]
                # misc["lemma_pred"] = apply_lemma_rule(token["form"], lemma_script)
                token["misc"] = misc


            else:
                # token["head"] = gov_dict.get(n_token+1, 'missing_gov')
                token["HEAD"] = head_chuliu
                token["UPOS"] = annotation_schema["uposs"][upos_index]
                token["FEATS"] = _featuresConllToJson(annotation_schema["feats"][feats_index])
                lemma_script = annotation_schema["lemma_scripts"][lemma_script_index]
                token["LEMMA"] = apply_lemma_rule(token["form"], lemma_script)
                token["DEPREL"] = annotation_schema["deprels"][deprel_chuliu]
        return predicted_sentence_json


def get_index(label: str, mapping: Dict) -> int:
    """
    label: a string that represent the label whose integer is required
    mapping: a dictionnary with a set of labels as keys and index integer as values

    return : index (int)
    """
    index = mapping.get(label, -1)

    if index == -1:
        index = mapping["none"]
        print(
            f"LOG: label '{label}' was not founded in the label2index mapping : ",
            mapping,
        )
    return index

