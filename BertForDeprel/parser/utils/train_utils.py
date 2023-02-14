import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch import optim, cuda, stack, gather
from .chuliu_edmonds_utils import chuliu_edmonds_one_root
import numpy as np
# Timing utility
from timeit import default_timer as timer
import time


def deprel_aligner_with_head(deprels_pred, heads_true):
    # print("KK heads_true", heads_true)
    # print("KK deprels_pred", deprels_pred.size())
    # print("KK heads_true", heads_true.size())
    heads_true = heads_true.unsqueeze(1).unsqueeze(2)
    # print("KK max", torch.max(heads_true))
    heads_true = heads_true.expand(-1, deprels_pred.size(1), -1, -1).clone()
    # print("KK deprels_pred", deprels_pred.size())
    # print("KK heads_true", heads_true.size())
    # print("KK heads_true", heads_true)
    # print("KK heads_true", deprels_pred)
    heads_true[heads_true< 0] = 0
    deprels_pred = gather(deprels_pred, 2, heads_true).squeeze(2)

    return deprels_pred

def compute_loss_head(heads_pred, heads_true, criterion):
    return criterion(heads_pred, heads_true)

def compute_loss_deprel(deprels_pred, deprels_true, heads_true, criterion):
    deprels_pred = deprel_aligner_with_head(deprels_pred, heads_true)

    return criterion(deprels_pred, deprels_true)

def compute_loss_poss(poss_pred, poss_true, criterion):
    return criterion(poss_pred.permute(0,2,1), poss_true)

def compute_acc_head(heads_pred, heads_true, eps=1e-10):
    mask = (heads_true!=int(heads_true[0][0]))
    good_head = float(sum(heads_true[mask] == heads_pred.max(dim=1)[1][mask]))
    total_head = float(sum(sum(mask))) + eps
    return good_head, total_head

def compute_acc_pos(poss_pred, poss_true, eps=1e-10):
    mask = (poss_true!=int(poss_true[0][0]))
    good_pos = float(sum(poss_true[mask] == poss_pred.max(dim=2)[1][mask]))
    total_pos = float(sum(sum(mask))) + eps
    return good_pos, total_pos

def compute_acc_deprel(deprels_pred, deprels_true, heads_true, eps=1e-10):
    mask = (heads_true!=int(heads_true[0][0]))
    deprels_pred = deprel_aligner_with_head(deprels_pred, heads_true)

    good_deprel = float(sum(deprels_pred.max(dim=1)[1][mask] == deprels_true[mask]))
    # TODO :
    # - find better formula for summing up
    total_deprel = float(sum(sum(mask))) + eps
    return good_deprel, total_deprel

def compute_LAS(heads_pred, deprels_pred, heads_true, deprels_true):
    mask = (heads_true!=int(heads_true[0][0]))
    deprels_pred = deprel_aligner_with_head(deprels_pred, heads_true)
    correct_head = heads_pred.max(dim=1)[1][mask] == heads_true[mask]
    correct_deprel = deprels_pred.max(dim=1)[1][mask] == deprels_true[mask]

    n_correct_LAS = sum(correct_head & correct_deprel).item()
    n_total = float(sum(sum(mask)))

    return n_correct_LAS, n_total

def compute_LAS_main_aux(heads_pred, deprels_main_pred, heads_true, deprels_main_true):
    mask = (heads_true!=int(heads_true[0][0]))
    deprels_main_pred = deprel_aligner_with_head(deprels_main_pred, heads_true)

    correct_head = heads_pred.max(dim=1)[1][mask] == heads_true[mask]
    correct_deprel_main = deprels_main_pred.max(dim=1)[1][mask] == deprels_main_true[mask]

    n_correct_LAS_main = sum(correct_head & correct_deprel_main).item()
    LAS_epoch = sum(correct_head & correct_deprel_main).item()
    n_total = float(sum(sum(mask)))

    return LAS_epoch, n_correct_LAS_main, n_total


def compute_LAS_chuliu_main_aux(heads_chuliu_pred, deprels_main_pred, heads_true, deprels_main_true):
    mask = (heads_true!=int(heads_true[0][0]))
    deprels_main_pred = deprel_aligner_with_head(deprels_main_pred, heads_true)

    correct_head = heads_chuliu_pred[mask] == heads_true[mask]
    correct_deprel_main = deprels_main_pred.max(dim=1)[1][mask] == deprels_main_true[mask]

    n_correct_LAS_main = sum(correct_head & correct_deprel_main).item()
    LAS_epoch = sum(correct_head & correct_deprel_main).item()
    n_total = float(sum(sum(mask)))

    return LAS_epoch, n_correct_LAS_main, n_total


def confusion_matrix(deprels_pred, deprels_true, heads_true, conf_matrix):
    mask = (heads_true!=int(heads_true[0][0]))
    # print("KK mask", mask)
    deprels_pred = deprel_aligner_with_head(deprels_pred, heads_true)

    trues = deprels_true[mask]
    preds = deprels_pred.max(dim=1)[1][mask]

    for p, t in zip(preds, trues):
        conf_matrix[p, t] += 1

    return conf_matrix


from time import time
ts = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
}

def train_epoch(model, n_epoch ,train_loader, args):
    device = args.device
    start = timer()
    model.train()
    for n_batch, (seq, subwords_start, attn_masks, idx_convertor, poss, heads, deprels_main) in enumerate(train_loader):
        args.optimizer.zero_grad()
        seq, attn_masks, heads_true, deprels_main_true, poss_true = seq.to(device), attn_masks.to(device), heads.to(device), deprels_main.to(device), poss.to(device)

        heads_pred, deprels_main_pred, poss_pred = model.forward(seq, attn_masks)
        
        loss_batch = 0.0
        loss_head_batch = compute_loss_head(heads_pred, heads_true, args.criterions['head'])
        loss_batch += loss_head_batch
        
        loss_deprels_main_batch = compute_loss_deprel(deprels_main_pred, deprels_main_true, heads_true.clone(), args.criterions['deprel'])
        loss_batch += loss_deprels_main_batch
        loss_poss_batch = compute_loss_poss(poss_pred, poss_true, args.criterions['pos'])
        loss_batch += loss_poss_batch
        
        loss_batch.backward()
        args.optimizer.step()

        print(
        f'Training: {100 * (n_batch + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds in epoch; loss : {loss_poss_batch:.2f} {loss_head_batch:.2f}',
        end='\r')
    

# TODO_LEMMA : add the lemma
def eval_epoch(model, eval_loader, args, n_epoch = -1):
    model.eval()
    device = args.device
    with torch.no_grad():
        loss_head_epoch = 0.0
        loss_deprel_main_epoch = 0.0
        loss_poss_epoch = 0
        good_head_epoch, total_head_epoch = 0.0, 0.0
        good_pos_epoch, total_pos_epoch = 0.0, 0.0
        good_deprel_main_epoch, total_deprel_main_epoch = 0.0, 0.0
        n_correct_LAS_epoch, n_correct_LAS_main_epoch,n_correct_LAS_aux_epoch, n_total_epoch = 0.0, 0.0, 0.0, 0.0
        n_correct_LAS_chuliu_epoch, n_correct_LAS_chuliu_main_epoch,n_correct_LAS_chuliu_aux_epoch, n_total_epoch = 0.0, 0.0, 0.0, 0.0
        
        conf_matrix = torch.zeros(args.n_labels_main, args.n_labels_main)
        for n_batch, (seq, subwords_start, attn_masks, idx_convertor, poss, heads, deprels_main) in enumerate(eval_loader):
            print(f"evaluation on the dataset ... {n_batch}/{len(eval_loader)}batches", end="\r")
            seq, attn_masks, heads_true, deprels_main_true, poss_true = seq.to(device), attn_masks.to(device), heads.to(device), deprels_main.to(device), poss.to(device)
            heads_pred, deprels_main_pred, poss_pred = model.forward(seq, attn_masks)
            # print("KK heads_pred", heads_pred.size())
            # print("KK deprels_main_pred", deprels_main_pred.size())
            # print("KK poss_pred", poss_pred.size())
            
            heads_pred, deprels_main_pred, poss_pred = heads_pred.detach(), deprels_main_pred.detach(), poss_pred.detach()

            chuliu_heads_pred = heads_true.clone()
            for i_vector, (heads_pred_vector, subwords_start_vector, idx_convertor_vector) in enumerate(zip(heads_pred, subwords_start, idx_convertor)):
                subwords_start_with_root = subwords_start_vector.clone()
                subwords_start_with_root[0] = True

                heads_pred_np = heads_pred_vector[:,subwords_start_with_root == 1][subwords_start_with_root == 1]
                heads_pred_np = heads_pred_np.cpu().numpy()
                
                chuliu_heads_vector = chuliu_edmonds_one_root(np.transpose(heads_pred_np, (1,0)))[1:]
                
                for i_token, chuliu_head_pred in enumerate(chuliu_heads_vector):
                    chuliu_heads_pred[i_vector, idx_convertor_vector[i_token+1]] = idx_convertor_vector[chuliu_head_pred]
                
            conf_matrix = confusion_matrix(deprels_main_pred, deprels_main_true, heads_true, conf_matrix)
            
            n_correct_LAS_batch, n_correct_LAS_main_batch, n_total_batch = \
                compute_LAS_main_aux(heads_pred, deprels_main_pred, heads_true, deprels_main_true)
            n_correct_LAS_chuliu_batch, n_correct_LAS_chuliu_main_batch, n_total_batch = \
                compute_LAS_chuliu_main_aux(chuliu_heads_pred, deprels_main_pred, heads_true, deprels_main_true)
            n_correct_LAS_epoch += n_correct_LAS_batch
            n_correct_LAS_chuliu_epoch += n_correct_LAS_chuliu_batch
            n_total_epoch += n_total_batch

            loss_head_batch = compute_loss_head(heads_pred, heads_true, args.criterions['head'])
            good_head_batch, total_head_batch = compute_acc_head(heads_pred, heads_true, eps=0)
            loss_head_epoch += loss_head_batch.item()
            good_head_epoch += good_head_batch
            total_head_epoch += total_head_batch
            
            loss_deprel_main_batch = compute_loss_deprel(deprels_main_pred, deprels_main_true, heads_true, args.criterions['deprel'])
            good_deprel_main_batch, total_deprel_main_batch = compute_acc_deprel(deprels_main_pred, deprels_main_true, heads_true, eps=0)
            loss_deprel_main_epoch += loss_deprel_main_batch.item()
            good_deprel_main_epoch += good_deprel_main_batch
            total_deprel_main_epoch += total_deprel_main_batch
            n_correct_LAS_main_epoch += n_correct_LAS_main_batch
            
            good_pos_batch, total_pos_batch = compute_acc_pos(poss_pred, poss_true, eps=0)
            good_pos_epoch += good_pos_batch
            total_pos_epoch += total_pos_batch

            loss_poss_batch = compute_loss_poss(poss_pred, poss_true, args.criterions['pos'])
            loss_poss_epoch += loss_poss_batch


        loss_head_epoch = loss_head_epoch/len(eval_loader)
        acc_head_epoch = good_head_epoch/total_head_epoch
        
        loss_deprel_main_epoch = loss_deprel_main_epoch/len(eval_loader)
        acc_deprel_main_epoch = good_deprel_main_epoch/total_deprel_main_epoch
        
        acc_pos_epoch = good_pos_epoch/total_pos_epoch

        LAS_epoch = n_correct_LAS_epoch/n_total_epoch
        LAS_chuliu_epoch = n_correct_LAS_chuliu_epoch/n_total_epoch
        LAS_main_epoch = n_correct_LAS_main_epoch/n_total_epoch


        loss_epoch = loss_head_epoch + loss_deprel_main_epoch + loss_poss_epoch 
        print("\nevaluation result: LAS={:.3f}; LAS_chuliu={:.3f}; loss_epoch={:.3f}; eval_acc_head={:.3f}; eval_acc_deprel = {:.3f}, eval_acc_pos = {:.3f}\n".format(
        LAS_epoch, LAS_chuliu_epoch, loss_epoch, LAS_main_epoch, acc_head_epoch, acc_pos_epoch))

    results = {
      "LAS_epoch": LAS_epoch,
      "LAS_chuliu_epoch": LAS_chuliu_epoch,
      "LAS_main_epoch": LAS_main_epoch,
      "acc_head_epoch": acc_head_epoch,
      "acc_deprel_main_epoch" : acc_deprel_main_epoch,
      "acc_pos_epoch": acc_pos_epoch,
      "loss_head_epoch": loss_head_epoch,
      "loss_deprel_main_epoch": loss_deprel_main_epoch,
      "loss_epoch": loss_epoch,
    }

    return results

def update_history(history, results, n_epoch, args):

    history.append([n_epoch, *results.values()])

    pd.DataFrame(
        history,
        columns=['n_epoch', *results.keys()]
          ).to_csv(args.model.replace(".pt", "_history.csv"), sep='\t', index=False)

    return history


def eisner(scores, mask):
    lens = mask.sum(1)
    batch_size, seq_len, _ = scores.shape
    scores = scores.permute(2, 1, 0)
    s_i = torch.full_like(scores, float('-inf'))
    s_c = torch.full_like(scores, float('-inf'))
    p_i = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_c = scores.new_zeros(seq_len, seq_len, batch_size).long()
    s_c.diagonal().fill_(0)

    for w in range(1, seq_len):
        n = seq_len - w
        starts = p_i.new_tensor(range(n)).unsqueeze(0)
        # ilr = C(i->r) + C(j->r+1)
        ilr = stripe(s_c, n, w) + stripe(s_c, n, w, (w, 1))
        # [batch_size, n, w]
        ilr = ilr.permute(2, 0, 1)
        il = ilr + scores.diagonal(-w).unsqueeze(-1)
        # I(j->i) = max(C(i->r) + C(j->r+1) + s(j->i)), i <= r < j
        il_span, il_path = il.max(-1)
        s_i.diagonal(-w).copy_(il_span)
        p_i.diagonal(-w).copy_(il_path + starts)
        ir = ilr + scores.diagonal(w).unsqueeze(-1)
        # I(i->j) = max(C(i->r) + C(j->r+1) + s(i->j)), i <= r < j
        ir_span, ir_path = ir.max(-1)
        s_i.diagonal(w).copy_(ir_span)
        p_i.diagonal(w).copy_(ir_path + starts)

        # C(j->i) = max(C(r->i) + I(j->r)), i <= r < j
        cl = stripe(s_c, n, w, (0, 0), 0) + stripe(s_i, n, w, (w, 0))
        cl_span, cl_path = cl.permute(2, 0, 1).max(-1)
        s_c.diagonal(-w).copy_(cl_span)
        p_c.diagonal(-w).copy_(cl_path + starts)
        # C(i->j) = max(I(i->r) + C(r->j)), i < r <= j
        cr = stripe(s_i, n, w, (0, 1)) + stripe(s_c, n, w, (1, w), 0)
        cr_span, cr_path = cr.permute(2, 0, 1).max(-1)
        s_c.diagonal(w).copy_(cr_span)
        s_c[0, w][lens.ne(w)] = float('-inf')
        p_c.diagonal(w).copy_(cr_path + starts + 1)

    predicts = []
    p_c = p_c.permute(2, 0, 1).cpu()
    p_i = p_i.permute(2, 0, 1).cpu()
    for i, length in enumerate(lens.tolist()):
        heads = p_c.new_ones(length + 1, dtype=torch.long)
        backtrack(p_i[i], p_c[i], heads, 0, length, True)
        predicts.append(heads.to(mask.device))

    return predicts
    # return pad_sequence(predicts, True)


def backtrack(p_i, p_c, heads, i, j, complete):
    if i == j:
        return
    if complete:
        r = p_c[i, j]
        backtrack(p_i, p_c, heads, i, r, False)
        backtrack(p_i, p_c, heads, r, j, True)
    else:
        r, heads[j] = p_i[i, j], i
        i, j = sorted((i, j))
        backtrack(p_i, p_c, heads, i, r, True)
        backtrack(p_i, p_c, heads, j, r + 1, True)


def stripe(x, n, w, offset=(0, 0), dim=1):
    r'''Returns a diagonal stripe of the tensor.
    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.
    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    '''
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)
      
