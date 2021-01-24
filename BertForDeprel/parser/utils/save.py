import torch
import json



def save_meta_model(model, n_epoch, eval_LAS_best, args):
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
        'state_dict': model.state_dict(),
        'eval_LAS_best': eval_LAS_best,
        'optimizer' : args.optimizer.state_dict(),
        # **vars(args),
        "args": args
        # 'i2drm':i2drm,
        # 'drm2i':drm2i,
        # 'i2dra':i2dra,
        # 'dra2i':dra2i,
              }
    torch.save(state, args.model)

    # print("**vars(args)", type(vars(args)))
    # for key, value in vars(args).items():
    #   print(key, str(value))

# def load_meta_model():
#     checkpoint = torch.load(args.model, map_location=torch.device('cpu'))

#     # because we saved the model with nn.Dataparralel, we need to change the state_dict keys
#     new_state_dict = OrderedDict()
#     for k, v in checkpoint['state_dict'].items():
#         name = k.replace("module.","")
#         new_state_dict[name] = v

#     model.load_state_dict(new_state_dict)
#     return 





