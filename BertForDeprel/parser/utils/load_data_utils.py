import conllu
from torch.utils.data import Dataset
from torch import tensor


class ConlluDataset(Dataset):
    # def __init__(self, path_file, tokenizer, bert_language="english", args.maxlen = 256, list_deprel_main=None, list_deprel_aux=None, separate_deprel=True, exclude_punc=False):
    def __init__(self, path_file, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

        # self.separate_deprel = args.separate_deprel
        self.separate_deprel = True

        self.CLS_token_id = tokenizer.cls_token_id
        self.SEP_token_id = tokenizer.sep_token_id


        # Load all the sequences from the file
        # TODO : make a generator
        with open(path_file, 'r') as infile:
            self.sequences = conllu.parse(infile.read())

#         infile = open(path_file, 'r')
#         self.sequences = conllu.parse_incr(infile)

        # self.primary_relation_only = primary_relation_only

        # if not self.args.list_deprel_main:
            # self.list_deprel_main, self.list_deprel_aux = create_deprel_lists(path_file)

        # print("list_deprel_main", self.list_deprel_main)
        # print("list_deprel_aux", self.list_deprel_aux)

        # Make the Dependency Relation to Index dictionary if not given in the init


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
        return(len(self.sequences))

    def __getitem__(self, index):
        return(self._get_processed(self.sequences[index]))

    def _mount_dr2i(self, list_deprel):
        i2dr = {}
        dr2i = {}

        for idx, deprel in enumerate(list_deprel):
            i2dr[idx] = deprel
            dr2i[deprel] = idx


        return dr2i, i2dr


    def _mount_pos2i(self, list_pos):
        # list_pos = []
        # for sequence in self.sequences:
        #     for token in sequence:
        #         list_pos.append(token['upostag'])
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


        return l + [padding_value]*(self.args.maxlen-len(l))
    
    def _trunc(self, tensor):
        if len(tensor) >= self.args.maxlen:
            tensor = tensor[:self.args.maxlen-1]
        
        return tensor


    def _get_input(self, sequence):
        sequence_ids = [self.CLS_token_id]
        subwords_start = [-1]
        idx_convertor = [0]
        tokens_len = [1]
        
        for token in sequence:
            if type(token['id']) != int:
                # print(token['id'])
                continue

            form = ""
            # if self.args.increment_unicode:
            #     for character in token['form']:
            #         form += chr(ord(character) + 3000) 
            # else:
            form = token['form']
            token_ids = self.tokenizer.encode(form, add_special_tokens=False)
            idx_convertor.append(len(sequence_ids))
            tokens_len.append(len(token_ids))
            subword_start = [1] + [0] * (len(token_ids)-1)
            
            sequence_ids += token_ids
            subwords_start += subword_start
        
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
        skipped_tokens = 0
        for n_token, token in enumerate(sequence):
            if type(token['id']) != int:
                # print(token['id'])
                skipped_tokens += 1
                continue
            
            # if len(tokens_len) == n_token+1:
            #     print("sequence", sequence)
            #     print("tokens_len", tokens_len)
            token_len = tokens_len[n_token + 1- skipped_tokens]
            
            pos = [self.pos2i.get(token['upostag'], self.pos2i['none'])]  + [-1]*(token_len-1)
            head = [sum(tokens_len[:token['head']])] + [-1]*(token_len-1)
            deprel_main, deprel_aux = normalize_deprel(token['deprel'], split_deprel=self.args.split_deprel)
            deprel_main = [self.drm2i.get(deprel_main, self.drm2i['none'])] + [-1]*(token_len-1)
            
            poss += pos
            heads += head
            deprels_main += deprel_main
            
            if self.args.split_deprel: 
                deprel_aux = [self.dra2i.get(deprel_aux, self.dra2i['none'])] + [-1]*(token_len-1)
                deprels_aux += deprel_aux
        # try:
        # except:
        #     print(sequence)
        #     print(sequence.metadata)
        heads = self._trunc(heads)
        deprels_main = self._trunc(deprels_main)
        poss = self._trunc(poss)
        
        poss = tensor(self._pad_list(poss, -1))
        heads = tensor(self._pad_list(heads, -1))
        deprels_main = tensor(self._pad_list(deprels_main, -1))

        heads[heads==-1] = self.args.maxlen - 1
        heads[heads>= self.args.maxlen-1] = self.args.maxlen - 1
        
        if self.args.split_deprel:
            deprel_aux = self._trunc(deprel_aux)
            deprels_aux = tensor(self._pad_list(deprels_aux, -1))
        
        if not self.args.punct:
            is_punc_tensor = [deprels_main==self.drm2i["punct"]]
            heads[is_punc_tensor] = self.args.maxlen - 1
            deprels_main[is_punc_tensor] = -1

            if self.args.split_deprel:
                deprels_aux[is_punc_tensor] = -1

        if not self.args.split_deprel:    
            deprels_aux = deprels_main.clone()
        
        return poss, heads, deprels_main, deprels_aux

    
    def _get_processed(self, sequence):
        sequence_ids, subwords_start, attn_masks, idx_convertor, token_lens = self._get_input(sequence)
        
        if self.args.mode == 'predict':
            return sequence_ids, subwords_start, attn_masks, idx_convertor
        
        else:
            poss, heads, deprels_main, deprels_aux = self._get_output(sequence, token_lens)
            
            return sequence_ids, subwords_start, attn_masks, idx_convertor, poss, heads, deprels_main, deprels_aux

def normalize_deprel(deprel, split_deprel):
    # change for taking only before @
    deprel = deprel.replace("@", ":")
    if split_deprel:
        deprels = deprel.split(":")
        deprel_main = deprels[0]
        if len(deprels) > 1:
            deprel_aux = deprels[1]
        else:
            deprel_aux = 'none'

        return deprel_main, deprel_aux

    else :
        return deprel, 'none'


def create_deprel_lists(*paths, split_deprel):
    print(paths)
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            result = conllu.parse(infile.read())

        list_deprel_main = []
        list_deprel_aux = []
        for sequence in result:
            for token in sequence:
                deprel_main, deprel_aux = normalize_deprel(token['deprel'], split_deprel=split_deprel)
                list_deprel_main.append(deprel_main)
                list_deprel_aux.append(deprel_aux)
    
    list_deprel_main.append('none')
    list_deprel_aux.append('none')
    list_deprel_main = sorted(set(list_deprel_main))
    list_deprel_aux = sorted(set(list_deprel_aux))
    return list_deprel_main, list_deprel_aux


def create_pos_list(*paths):
    list_pos = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as infile:
            result = conllu.parse(infile.read())

        for sequence in result:
            for token in sequence:
                list_pos.append(token['upostag'])
    list_pos.append('none')
    list_pos = sorted(set(list_pos))
    return list_pos