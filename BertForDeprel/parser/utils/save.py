import torch
import json
from .model_utils import BertForDeprel

from time import time

ts = {
    1: 0,
    2: 0,
}

def save_meta_model(model: BertForDeprel, n_epoch, eval_LAS_best, args):
    t = time()
    model.llm_layer.save_adapter("/home/kirian/adapter_saved_test.pt", "tagger")
    ts[1] += round(time() - t, 3)
    state = {
        "n_epoch" : n_epoch,
        # "maxlen": maxlen,
        # "batch_size": params["batch_size"],
        # 'n_labels_main': n_labels_main,
        # 'n_labels_aux': n_labels_aux,
        # "list_deprel_main": list_deprel_main,
        # "list_deprel_aux": list_deprel_aux,
        # 'exclude_punc': exclude_punc,
        # "bert_language": bert_language,
        # 'state_dict': model.state_dict(),
        # 'llm_layer_state': model.llm_layer.state_dict(),
        'tagger_layer_state': model.tagger_layer.state_dict(),
        'eval_LAS_best': eval_LAS_best,
        # 'optimizer' : args.optimizer.state_dict(),
        # **vars(args),
        # "args": args
        # 'i2drm':i2drm,
        # 'drm2i':drm2i,
        # 'i2dra':i2dra,
        # 'dra2i':dra2i,
              }
    t = time()
    torch.save(state, args.name_model)
    ts[2] += round(time() - t, 3)
    print("\n ts", ts)



def save_model_weights(self, ckpt_fpath, epoch):
    trainable_weight_names = [n for n, p in self.model_parameters if p.requires_grad]
    state = {
        'adapters': {},
        'epoch': epoch
    }
    for k, v in self._embedding_layers.state_dict().items():
        if k in trainable_weight_names:
            state['adapters'][k] = v
    if self._task == 'tokenize':
        for k, v in self._tokenizer.state_dict().items():
            if k in trainable_weight_names:
                state['adapters'][k] = v
    elif self._task == 'posdep':
        for k, v in self._tagger.state_dict().items():
            if k in trainable_weight_names:
                state['adapters'][k] = v
    elif self._task == 'ner':
        for k, v in self._ner_model.state_dict().items():
            if k in trainable_weight_names:
                state['adapters'][k] = v

    torch.save(state, ckpt_fpath)
    print('Saving adapter weights to ... {} ({:.2f} MB)'.format(ckpt_fpath,
                                                                os.path.getsize(ckpt_fpath) * 1. / (1024 * 1024)))

    # print("**vars(args)", type(vars(args)))
    # for key, value in vars(args).items():
    #   print(key, str(value))

# def load_meta_model():
#     checkpoint = torch.load(args.name_model, map_location=torch.device('cpu'))

#     # because we saved the model with nn.Dataparralel, we need to change the state_dict keys
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint['state_dict'].items():
#         name = k.replace("module.","")
#         new_state_dict[name] = v

#     model.load_state_dict(new_state_dict)
#     return 





