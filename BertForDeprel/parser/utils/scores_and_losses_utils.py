import pandas as pd
from torch import gather, Tensor
from torch.nn import CrossEntropyLoss


def deprel_aligner_with_head(deprels_pred: Tensor, heads_true: Tensor):
    heads_true = heads_true.unsqueeze(1).unsqueeze(2)
    heads_true = heads_true.expand(-1, deprels_pred.size(1), -1, -1).clone()
    heads_true[heads_true< 0] = 0
    deprels_pred = gather(deprels_pred, 2, heads_true).squeeze(2)

    return deprels_pred

def compute_loss_head(heads_pred: Tensor, heads_true: Tensor, criterion: CrossEntropyLoss):
    return criterion.forward(heads_pred, heads_true)

def compute_loss_deprel(deprels_pred: Tensor, deprels_true: Tensor, heads_true, criterion: CrossEntropyLoss):
    deprels_pred = deprel_aligner_with_head(deprels_pred, heads_true)
    return criterion.forward(deprels_pred, deprels_true)

def compute_loss_poss(poss_pred: Tensor, poss_true: Tensor, criterion: CrossEntropyLoss):
    return criterion.forward(poss_pred.permute(0,2,1), poss_true)

def compute_acc_head(heads_pred: Tensor, heads_true: Tensor, eps=1e-10):
    mask = (heads_true!=int(heads_true[0][0]))
    good_head = float(sum(heads_true[mask] == heads_pred.max(dim=1)[1][mask]))
    total_head = float(sum(sum(mask))) + eps
    return good_head, total_head

def compute_acc_upos(poss_pred: Tensor, poss_true: Tensor, eps=1e-10):
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

def compute_LAS(heads_pred: Tensor, deprels_main_pred: Tensor, heads_true: Tensor, deprels_main_true: Tensor):
    """Labled Attachment Score measures the accuracy of labeled dependency edges.
    Returns (number correct, total) indicating the number of correctly labeled dependency edges
    and the total number of edges predicted.
    """
    # TODO what is being masked here? Is this ignoring a sentence token, perhaps? Hmm, no the SEP token is last, not first.
    mask = (heads_true!=int(heads_true[0][0]))
    deprels_main_pred = deprel_aligner_with_head(deprels_main_pred, heads_true)

    correct_head = heads_pred.max(dim=1)[1][mask] == heads_true[mask]
    correct_deprel_main = deprels_main_pred.max(dim=1)[1][mask] == deprels_main_true[mask]

    n_correct_LAS_main = sum(correct_head & correct_deprel_main).item()
    n_total = float(sum(sum(mask)))

    return n_correct_LAS_main, n_total


def compute_LAS_chuliu(heads_chuliu_pred: Tensor, deprels_main_pred: Tensor, heads_true: Tensor, deprels_main_true: Tensor):
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
    deprels_pred = deprel_aligner_with_head(deprels_pred, heads_true)

    trues = deprels_true[mask]
    preds = deprels_pred.max(dim=1)[1][mask]

    for p, t in zip(preds, trues):
        conf_matrix[p, t] += 1

    return conf_matrix


def update_history(history, results, n_epoch, args):

    history.append([n_epoch, *results.values()])

    pd.DataFrame(
        history,
        columns=['n_epoch', *results.keys()]
          ).to_csv(args.name_model.replace(".pt", "_history.csv"), sep='\t', index=False)

    return history


