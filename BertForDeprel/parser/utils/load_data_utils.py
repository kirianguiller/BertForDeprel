from typing import Dict, List, Any
import conllu
from torch.utils.data import Dataset
from torch import tensor
from transformers import RobertaTokenizer
from .types import ModelParams_T

class ConlluDataset(Dataset):
    def __init__(self, path_file: str, tokenizer: RobertaTokenizer, model_params: ModelParams_T, run_mode: str):
        self.tokenizer = tokenizer
        self.run_mode = run_mode
        self.model_params = model_params

        self.CLS_token_id = tokenizer.cls_token_id
        self.SEP_token_id = tokenizer.sep_token_id

        # Load all the sequences from the file
        # TODO : make a generator
        with open(path_file, "r") as infile:
            self.sequences = conllu.parse(infile.read())

        self.dep2i, self.i2dep = self._compute_labels2i(self.model_params["annotation_schema"]["deprels"])
        self.pos2i, self.i2pos = self._compute_labels2i(self.model_params["annotation_schema"]["uposs"])
        # self.lemma_script2i, self.i2lemma_script = self._compute_labels2i(self.args.list_lemma_script)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self._get_processed(self.sequences[index])

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

    def _get_input(self, sequence):
        sequence_ids = [self.CLS_token_id]
        subwords_start = [-1]
        idx_convertor = [0]
        tokens_len = [1]

        for token in sequence:
            if type(token["id"]) != int:
                continue

            form = ""
            form = token["form"]
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

        sequence_ids = tensor(sequence_ids)
        subwords_start = tensor(subwords_start)
        idx_convertor = tensor(idx_convertor)
        attn_masks = tensor([int(token_id > 0) for token_id in sequence_ids])
        return sequence_ids, subwords_start, attn_masks, idx_convertor, tokens_len

    def _get_output(self, sequence, tokens_len):
        poss = [-1]
        heads = [-1]
        deprels_main = [-1]
        skipped_tokens = 0
        
        for n_token, token in enumerate(sequence):
            if type(token["id"]) != int:
                skipped_tokens += 1
                continue

            token_len = tokens_len[n_token + 1 - skipped_tokens]

            pos = [get_index(token["upostag"], self.pos2i)] + [-1] * (token_len - 1)
            
            head = [sum(tokens_len[: token["head"]])] + [-1] * (token_len - 1)
            deprel_main = token["deprel"]

            deprel_main = [get_index(deprel_main, self.dep2i)] + [-1] * (
                token_len - 1
            )
            # Example of what we have for a token of 2 subtokens
            # form = ["eat", "ing"]
            # pos = [4, -1]
            # head = [2, -1]
            # lemma_script = [3424, -1]
            # token_len = 2
            poss += pos
            heads += head
            deprels_main += deprel_main
        heads = self._trunc(heads)
        deprels_main = self._trunc(deprels_main)
        poss = self._trunc(poss)

        poss = tensor(poss)
        heads = tensor(heads)
        deprels_main = tensor(deprels_main)

        return poss, heads, deprels_main

    def _get_processed(self, sequence):
        (
            sequence_ids,
            subwords_start,
            attn_masks,
            idx_convertor,
            token_lens,
        ) = self._get_input(sequence)

        if self.run_mode == "predict":
            return sequence_ids, subwords_start, attn_masks, idx_convertor

        else:
            poss, heads, deprels_main = self._get_output(
                sequence, token_lens
            )

            return (
                sequence_ids,
                subwords_start,
                attn_masks,
                idx_convertor,
                poss,
                heads,
                deprels_main,
            )
    def collate_fn(self, sentences):
        max_sentence_length = max([len(sentence[0]) for sentence in sentences])
        sequence_ids   = tensor([self._pad_list(sentence[0].tolist(),  0, max_sentence_length) for sentence in sentences])
        subwords_start = tensor([self._pad_list(sentence[1].tolist(), -1, max_sentence_length) for sentence in sentences])
        attn_masks     = tensor([self._pad_list(sentence[2].tolist(),  0, max_sentence_length) for sentence in sentences])
        idx_convertor  = tensor([self._pad_list(sentence[3].tolist(), -1, max_sentence_length) for sentence in sentences])
        
        if self.run_mode == "train":
            poss           = tensor([self._pad_list(sentence[4].tolist(), -1, max_sentence_length) for sentence in sentences])
            heads          = tensor([self._pad_list(sentence[5].tolist(), -1, max_sentence_length) for sentence in sentences])
            deprels_main   = tensor([self._pad_list(sentence[6].tolist(), -1, max_sentence_length) for sentence in sentences])
            return (
                    sequence_ids,
                    subwords_start,
                    attn_masks,
                    idx_convertor,
                    poss,
                    heads,
                    deprels_main,
                )
        else:
            return (
                    sequence_ids,
                    subwords_start,
                    attn_masks,
                    idx_convertor,
                )


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
