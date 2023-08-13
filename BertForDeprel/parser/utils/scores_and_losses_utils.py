from torch import Tensor, gather
from torch import sum as tsum
from torch.nn import CrossEntropyLoss

from .annotation_schema import DUMMY_ID


def _deprel_pred_for_heads(deprel_scores_pred: Tensor, heads_pred: Tensor):
    """
    Given the dependency relation label score predictions for all possible heads of
    each word and the list of predicted heads, return the scores with the head dimension
    removed, and just the scores for the labels on the predicted dependency-head arcs
    remaining.

    deprels_pred: tensor with 4 dimensions: (batch_len, n_class_deprel, seq_len,
    seq_len). Read indexing as [sentence_index][deprel_label_index][dependent_index]
    [head_index], with the last dimension containing the scores for each potential head
    of the dependent.

    heads_pred: tensor with two dimensions: (batch_len, seq_len). Read indexing as
    [sentence_index][dependent_index], with the value at each index being the predicted
    head index.

    returns: tensor of size (batch_len, n_class_deprel, seq_len). Read indexing as
    [sentence_index][deprel_label_index][dependent_index], with the value at each index
    being the score for the given dependency relation label for the dependency edge
    between a dependent and its predicted head (which was specified in heads_pred, and
    is not contained explicitly in the output tensor).

    See test case if it's still not clear what this function does.
    """
    # modify heads_true to have the same shape as deprels_pred
    # add two dimensions of size 1: (batch_size, 1, 1, seq_len)
    heads_pred = heads_pred.unsqueeze(1).unsqueeze(2)
    # expand to (batch_size, n_class_deprel, 1, seq_len)
    heads_pred = heads_pred.expand(-1, deprel_scores_pred.size(1), -1, -1).clone()
    heads_pred[heads_pred == DUMMY_ID] = 0
    # deprels_pred.shape after gather =  (batch_len, n_class_deprel, 1, sq_len)
    # deprels_pred.shape after squeeze =  (batch_len, n_class_deprel, seq_len)
    deprel_scores_pred = gather(deprel_scores_pred, 2, heads_pred).squeeze(2)

    return deprel_scores_pred


def compute_loss_head(
    heads_pred: Tensor, heads_true: Tensor, criterion: CrossEntropyLoss
):
    """
    See _deprel_pred_for_heads for deeper explanation of the shapes of these parameters.
    heads_pred: (batch_len, seq_len)
    heads_true: (batch_len, seq_len)
    criterion: the loss function to perform forward propagation with
    """
    return criterion.forward(heads_pred, heads_true)


def compute_loss_deprel(
    deprel_scores_pred: Tensor,
    deprels_true: Tensor,
    heads_true,
    criterion: CrossEntropyLoss,
):
    """
    See _deprel_pred_for_heads for deeper explanation of the shapes of these parameters.
    deprel_scores_pred: (batch_len, n_class_deprel, seq_len, seq_len)
    deprels_true: (batch_len, n_class_deprel, seq_len)
    heads_true: (batch_len, seq_len)
    criterion: the loss function to perform forward propagation with
    """
    deprels_pred = _deprel_pred_for_heads(deprel_scores_pred, heads_true)
    return criterion.forward(deprels_pred, deprels_true)


def compute_loss_class(
    class_pred: Tensor, class_true: Tensor, criterion: CrossEntropyLoss
):
    """
    This applies to any of the 1-dimensional classification tasks
    class_pred: (batch_len {index of sentence in batch}, seq_len {token index},
    num_classes {score assigned to each class})
    class_true: (batch_len {index of sentence in batch}, seq_len {index of true class
    label}})
    criterion: the loss function to perform forward propagation with
    """
    # CrossEntropyLoss expects `input` parameter to have the num_classes
    # (C) dimension second, so we have to permute here
    return criterion.forward(class_pred.permute(0, 2, 1), class_true)


def __sum_2d_tensor(tensor: Tensor):
    return float(tsum(tensor, dim=(0, 1)).item())


def __discrete_preds_from_scores(score_preds: Tensor, dim=1):
    (_values, indices) = score_preds.max(dim=dim)
    return indices


def __mask_from_dummies(heads_true: Tensor):
    # the dummy ID is used to indicate indices that aren't being predicted for. For
    # example, heads tensors contain dummies for tokens that don't begin a real word
    # (CLS, subwords, padding, etc.)
    return heads_true != DUMMY_ID


def compute_acc_head(head_scores_pred: Tensor, heads_true: Tensor):
    """
    head_scores_pred: (batch_len {index of sentence in batch}, seq_len {index of token
    whose head is being predicted}, seq_len {score of each other token being assigned as
    head}).
    heads_true: (batch_len, seq_len)
    returns correct, total (client must aggregate and compute accuracy)
    """
    mask = __mask_from_dummies(heads_true)
    total = __sum_2d_tensor(mask)
    heads_pred = __discrete_preds_from_scores(head_scores_pred)
    correct = float(sum(heads_true[mask] == heads_pred[mask]))
    return correct, total


def compute_acc_class(class_scores_pred: Tensor, class_true: Tensor):
    """This applies to any of the 1-dimensional classification tasks
    class_scores_pred: (batch_len, seq_len, n_class)
    class_true: (batch_len, seq_len)
    returns correct, total (client must aggregate and compute accuracy)
    """
    mask = __mask_from_dummies(class_true)
    total = __sum_2d_tensor(mask)
    classes_pred = __discrete_preds_from_scores(class_scores_pred, dim=2)
    correct = float(sum(class_true[mask] == classes_pred[mask]))
    return correct, total


def compute_acc_deprel(deprel_scores_pred, deprels_true, heads_true):
    """
    See _deprel_pred_for_heads for deeper explanation of the shapes of these parameters.
    deprel_scores_pred: (batch_len, n_class_deprel, seq_len, seq_len)
    deprels_true: (batch_len, n_class_deprel, seq_len)
    heads_true: (batch_len, seq_len)
    returns correct, total (client must aggregate and compute accuracy)
    """
    mask = __mask_from_dummies(heads_true)
    total = __sum_2d_tensor(mask)
    deprel_scores_pred_for_true_heads = _deprel_pred_for_heads(
        deprel_scores_pred, heads_true
    )

    deprels_pred = __discrete_preds_from_scores(deprel_scores_pred_for_true_heads)
    correct = float(sum(deprels_pred[mask] == deprels_true[mask]))
    return correct, total


def compute_LAS(
    head_scores_pred: Tensor,
    deprel_scores_pred: Tensor,
    heads_true: Tensor,
    deprels_true: Tensor,
):
    """Labled Attachment Score measures the accuracy of labeled dependency edges.
    Returns (number correct, total) indicating the number of correctly labeled
    dependency edges and the total number of edges predicted.
    head_scores_pred: (batch_len {index of sentence in batch}, seq_len {index of token
    whose head is being predicted}, seq_len {score of each other token being assigned as
    head}).
    deprels_pred: (batch_len, n_class_deprel, seq_len, seq_len)
    heads_true: (batch_len, seq_len)
    deprels_true: (batch_len, seq_len)
    """
    deprel_scores_pred_for_true_heads = _deprel_pred_for_heads(
        deprel_scores_pred, heads_true
    )

    mask = __mask_from_dummies(heads_true)
    total = __sum_2d_tensor(mask)
    heads_pred = __discrete_preds_from_scores(head_scores_pred)
    deprels_pred = __discrete_preds_from_scores(deprel_scores_pred_for_true_heads)

    correct_head = heads_pred[mask] == heads_true[mask]
    correct_deprel = deprels_pred[mask] == deprels_true[mask]

    correct_LAS = tsum(correct_head & correct_deprel).item()

    return correct_LAS, total


def compute_LAS_chuliu(
    heads_pred: Tensor,
    deprel_scores_pred: Tensor,
    heads_true: Tensor,
    deprels_true: Tensor,
):
    """The difference between this and compute_LAS is only in how the head predictions
    are provided; this function accepts predicted head indices, while the other accepts
    scores over all possible heads for each dependent.
    heads_pred: tensor of dimensions (batch_len, seq_len) containing the predicted head
    index for each dependent."""
    deprel_scores_pred_for_true_heads = _deprel_pred_for_heads(
        deprel_scores_pred, heads_true
    )

    mask = __mask_from_dummies(heads_true)
    total = __sum_2d_tensor(mask)
    deprels_pred = __discrete_preds_from_scores(deprel_scores_pred_for_true_heads)

    correct_head = heads_pred[mask] == heads_true[mask]
    correct_deprel = deprels_pred[mask] == deprels_true[mask]

    correct_LAS = tsum(correct_head & correct_deprel).item()

    return correct_LAS, total


def confusion_matrix(deprel_scores_pred, deprels_true, heads_true, conf_matrix):
    mask = __mask_from_dummies(heads_true)
    deprel_scores_pred_for_true_heads = _deprel_pred_for_heads(
        deprel_scores_pred, heads_true
    )

    deprel_preds = __discrete_preds_from_scores(deprel_scores_pred_for_true_heads)

    for p, t in zip(deprel_preds[mask], deprels_true[mask]):
        conf_matrix[p, t] += 1

    return conf_matrix
