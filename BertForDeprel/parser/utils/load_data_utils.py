from typing import Dict
import conllu
from torch.utils.data import Dataset
from torch import tensor


class ConlluDataset(Dataset):
    def __init__(self, path_file, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

        # self.separate_deprel = args.separate_deprel
        self.separate_deprel = True

        self.CLS_token_id = tokenizer.cls_token_id
        self.SEP_token_id = tokenizer.sep_token_id

        # Load all the sequences from the file
        # TODO : make a generator
        with open(path_file, "r") as infile:
            self.sequences = conllu.parse(infile.read())    # sequences is a Tokenlist object

        self.drm2i, self.i2drm = self._mount_dr2i(self.args.list_deprel_main)

        self.pos2i, self.i2pos = self._mount_pos2i(self.args.list_pos)

        print("drm2i", self.drm2i)
        print("pos2i", self.pos2i)
        self.n_labels_main = len(self.drm2i)

        if self.args.split_deprel:
            self.dra2i, self.i2dra = self._mount_dr2i(self.args.list_deprel_aux)
            print("dra2i", self.dra2i)
            self.n_labels_aux = len(self.dra2i)

    def __len__(self):
        return len(self.sequences)  

    def __getitem__(self, index):
        return self._get_processed(self.sequences[index])

    def _mount_dr2i(self, list_deprel):
        i2dr = {}
        dr2i = {}

        for idx, deprel in enumerate(list_deprel):
            i2dr[idx] = deprel
            dr2i[deprel] = idx

        return dr2i, i2dr

    def _mount_pos2i(self, list_pos):
        sorted_set_pos = sorted(set(list_pos))

        pos2i = {}
        i2pos = {}

        for i, pos in enumerate(sorted_set_pos):
            pos2i[pos] = i
            i2pos[i] = pos

        self.list_pos = sorted_set_pos

        return pos2i, i2pos

    def _pad_list(self, l, padding_value):
        if len(l) > self.args.maxlen:
            print(l, len(l))
            raise Exception("The sequence is bigger than the size of the tensor")

        return l + [padding_value] * (self.args.maxlen - len(l))

    def _trunc(self, tensor):
        if len(tensor) >= self.args.maxlen:
            tensor = tensor[: self.args.maxlen - 1]

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
            token_ids = self.tokenizer.encode(form, add_special_tokens=False)   # Converts a string to a sequence of ids.
            idx_convertor.append(len(sequence_ids))
            tokens_len.append(len(token_ids))
            subword_start = [1] + [0] * (len(token_ids) - 1)    # if (len(token_ids) - 1)=2, return [1,0,0]

            sequence_ids += token_ids
            subwords_start += subword_start # records all subwords starts in a list

        # Cut, if beyond the max length
        sequence_ids = self._trunc(sequence_ids)
        subwords_start = self._trunc(subwords_start)
        idx_convertor = self._trunc(idx_convertor)

        sequence_ids = sequence_ids + [self.SEP_token_id]

        sequence_ids = tensor(self._pad_list(sequence_ids, 0))
        subwords_start = tensor(self._pad_list(subwords_start, -1))
        idx_convertor = tensor(self._pad_list(idx_convertor, -1))
        attn_masks = tensor([int(token_id > 0) for token_id in sequence_ids])

        return sequence_ids, subwords_start, attn_masks, idx_convertor, tokens_len

    def _get_output(self, sequence, tokens_len):
        poss = [-1]
        heads = [-1]
        deprels_main = [-1]
        deprels_aux = [-1]
        # TODO_LEMMA : add the lemma edit script list for each sequence
        # lemma_scripts = [-1] # how to initialize ?
        # SES is the shortest edit script, each SES of a token is a string
        SESs = [""]
        # Lemma_rules stores all rules of tokens that transform form to lemma 
        lemma_rules = ["None"]
        skipped_tokens = 0
        
        for n_token, token in enumerate(sequence):
            if type(token["id"]) != int:
                skipped_tokens += 1
                continue

            token_len = tokens_len[n_token + 1 - skipped_tokens]


            pos = [get_index(token["upostag"], self.pos2i)] + [-1] * (token_len - 1)
            head = [sum(tokens_len[: token["head"]])] + [-1] * (token_len - 1)
            deprel_main, deprel_aux = normalize_deprel(
                token["deprel"], split_deprel=self.args.split_deprel
            )
            deprel_main = [get_index(deprel_main, self.drm2i)] + [-1] * (
                token_len - 1
            )
            # TODO_LEMMA : find the lemma_script for the token , and then append it to the ...
            # ... lemma_scripts (list for the sequence)
            # lemma_script = ?????
            # lemma_scripts += lemma_script

            # Example of what we have for a token of 2 subtokens
            # form = ["eat", "ing"]
            # pos = [4, -1]
            # head = [2, -1]
            # lemma_script = [3424, -1]
            # token_len = 2
            SES = short_edit_script(token["form"], token["lemma"], allow_copy=False)    # If need?
            lemma_rule = gen_lemma_rule(token["form"], token["lemma"])

            poss += pos
            heads += head
            deprels_main += deprel_main
            SESs.append(SES)
            lemma_rules.append(lemma_rule) 
            
            if self.args.split_deprel:
                deprel_aux = [get_index(deprel_aux, self.dra2i)] + [-1] * (
                    token_len - 1
                )
                deprels_aux += deprel_aux
        heads = self._trunc(heads)
        deprels_main = self._trunc(deprels_main)
        poss = self._trunc(poss)

        poss = tensor(self._pad_list(poss, -1))
        heads = tensor(self._pad_list(heads, -1))
        deprels_main = tensor(self._pad_list(deprels_main, -1))
        # TODO_LEMMA : pad the lemma_scripts list (with which values ? empty strings ?)

        heads[heads == -1] = self.args.maxlen - 1
        heads[heads >= self.args.maxlen - 1] = self.args.maxlen - 1

        if self.args.split_deprel:
            deprel_aux = self._trunc(deprel_aux)
            deprels_aux = tensor(self._pad_list(deprels_aux, -1))

        if not self.args.punct:
            is_punc_tensor = [deprels_main == self.drm2i["punct"]]
            heads[is_punc_tensor] = self.args.maxlen - 1
            deprels_main[is_punc_tensor] = -1

            if self.args.split_deprel:
                deprels_aux[is_punc_tensor] = -1

        if not self.args.split_deprel:
            deprels_aux = deprels_main.clone()


        # TODO_LEMMA : don't forget to return the lemma_scripts 
        return poss, heads, deprels_main, deprels_aux, SESs

    def _get_processed(self, sequence):
        (
            sequence_ids,
            subwords_start,
            attn_masks,
            idx_convertor,
            token_lens,
        ) = self._get_input(sequence)

        if self.args.mode == "predict":
            return sequence_ids, subwords_start, attn_masks, idx_convertor

        else:
            # TODO_LEMMA : don't forget to return the lemma_scripts 
            poss, heads, deprels_main, deprels_aux, SES = self._get_output(
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
                deprels_aux,
                SES,
            )


def normalize_deprel(deprel, split_deprel): # Same thing in compute_annotation_schema, doesn't apply
    if split_deprel:
        deprels = deprel.split(":")
        deprel_main = deprels[0]
        if len(deprels) > 1:
            deprel_aux = deprels[1]
        else:
            deprel_aux = "none"

        return deprel_main, deprel_aux

    else:
        return deprel, "none"


def create_deprel_lists(*paths, split_deprel):  # Same thing in compute_annotation_schema, doesn't apply
    print(paths)
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            result = conllu.parse(infile.read())

        list_deprel_main = []
        list_deprel_aux = []
        for sequence in result:
            for token in sequence:
                deprel_main, deprel_aux = normalize_deprel(
                    token["deprel"], split_deprel=split_deprel
                )
                list_deprel_main.append(deprel_main)
                list_deprel_aux.append(deprel_aux)

    list_deprel_main.append("none")
    list_deprel_aux.append("none")
    list_deprel_main = sorted(set(list_deprel_main))
    list_deprel_aux = sorted(set(list_deprel_aux))
    return list_deprel_main, list_deprel_aux


def create_deprel_lists2(*paths):   # Same thing in compute_annotation_schema
    print(paths)
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            result = conllu.parse(infile.read())

        deprels = []
        for sequence in result:
            for token in sequence:
                deprels.append(token["deprel"])
    return set(deprels)


def create_pos_list(*paths):    # Same thing in compute_annotation_schema, doesn't apply
    list_pos = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            result = conllu.parse(infile.read())

        for sequence in result:
            for token in sequence:
                list_pos.append(token["upostag"])
    list_pos.append("none")
    list_pos = sorted(set(list_pos))
    return list_pos


def create_annotation_schema(*paths):   # Same thing in compute_annotation_schema, doesn't apply
    annotation_schema = {}

    deprels = create_deprel_lists2(*paths)

    mains, auxs, deeps = [], [], []

    for deprel in deprels:
        if deprel.count("@") == 1:
            deprel, deep = deprel.split("@")
            deeps.append(deep)
        if deprel.count(":") == 1:
            deprel, aux = deprel.split(":")
            auxs.append(aux)

        if (":" not in deprel) and ("@" not in deprel):
            mains.append(deprel)

    auxs.append("none")
    deeps.append("none")

    annotation_schema["deprel"] = list(set(deprels))
    annotation_schema["main"] = list(set(mains))
    annotation_schema["aux"] = list(set(auxs))
    annotation_schema["deep"] = list(set(deeps))

    upos = create_pos_list(*paths)

    annotation_schema["upos"] = upos
    print(annotation_schema)
    return annotation_schema


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



# TODO_LEMMA : create a function that take the form and the lemma of a token, and return the ...
# ... lemma script
# def name_this_function_properly(form, token):
#    return lemma_script
def short_edit_script(source, target, allow_copy=False):
    """
    Finds the minimum edit script to transform the source to the target
    source->form, target->lemma
    """
    a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(0, len(source) + 1):
        for j in range(0, len(target) + 1):
            if i == 0 and j == 0:
                a[i][j] = (0, "")
            else:
                if allow_copy and i and j and source[i - 1].lower() == target[j - 1] and a[i-1][j-1][0] < a[i][j][0]:
                    if source[i - 1] == target[j - 1]:
                        a[i][j] = (a[i-1][j-1][0], a[i-1][j-1][1] + "→")
                    else:
                        a[i][j] = (a[i-1][j-1][0], a[i-1][j-1][1] + "↓") 
                if i and a[i-1][j][0] < a[i][j][0]:
                    a[i][j] = (a[i-1][j][0] + 1, a[i-1][j][1] + "-")
                if j and a[i][j-1][0] < a[i][j][0]:
                    a[i][j] = (a[i][j-1][0] + 1, a[i][j-1][1] + "+" + target[j - 1])
    return a[-1][-1][1]
    return a[-1][-1][1] # Return the last one, the SES

def gen_lemma_rule(form, lemma, allow_copy=True):
    """
    Generates a lemma rule to transform the source to the target
    """
    form = form.lower()

    previous_case = -1
    lemma_casing = ""
    for i, c in enumerate(lemma):
        case = "↑" if c.lower() != c else "↓"
        if case != previous_case:
            lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case, i if i <= len(lemma) // 2 else i - len(lemma))
        previous_case = case
    lemma = lemma.lower()

    best, best_form, best_lemma = 0, 0, 0
    for l in range(len(lemma)):
        for f in range(len(form)):
            cpl = 0
            while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl] == lemma[l + cpl]: cpl += 1
            if cpl > best:
                best = cpl
                best_form = f   # form从这开始，有best个字母的字符串和lemma一样
                best_lemma = l

    rule = ""
    if not best:
        rule += "a" + lemma
    else:
        rule += "{}->{}¦{}|".format(
            form[:best_form],
            lemma[:best_lemma],
            short_edit_script(form[:best_form], lemma[:best_lemma], allow_copy=True),
        )

        rule += "{}->{}¦{}".format(
            form[best_form + best:],
            lemma[best_lemma + best:],
            short_edit_script(form[best_form + best:], lemma[best_lemma + best:], allow_copy=True),
        )
    return rule