import torch

from BertForDeprel.parser.utils.annotation_schema import DUMMY_ID
from BertForDeprel.parser.utils.scores_and_losses_utils import _deprel_pred_for_heads


def test_deprel_pred_for_heads():
    batch_len = 3
    n_class_deprel = 2
    seq_len = 5

    # Create deprels_pred of size (batch_len, n_class_deprel, seq_len, seq_len)
    # Fill with dummy data that's easy to inspect
    relation_scores = torch.tensor(range(1, seq_len + 1))
    deprel_scores = torch.stack([relation_scores * i for i in range(1, seq_len + 1)])
    deprel_scores_per_relation = torch.stack(
        [deprel_scores * i for i in range(1, n_class_deprel + 1)]
    )
    deprel_scores_pred = torch.stack(
        [deprel_scores_per_relation * i for i in range(1, batch_len + 1)]
    )
    assert deprel_scores_pred.size() == (batch_len, n_class_deprel, seq_len, seq_len)
    # read the output like this: for each sentence in the batch, for each potential
    # deprel class label, for each potential dependent, the score for each potential
    # head
    # print(deprels_pred)

    # read: in the second sentence, indices 0 and 4 have no predicted head, and indices
    # 1, 2, and 3 have predicted heads 2, 3, and 4, respectively.
    heads_pred = torch.tensor(
        [
            [DUMMY_ID, 1, 2, DUMMY_ID, 3],
            [DUMMY_ID, 2, 3, 4, DUMMY_ID],
            [DUMMY_ID, 4, DUMMY_ID, 2, 3],
        ]
    )
    assert heads_pred.size() == (batch_len, seq_len)

    # The return value removes the head dimension, since we specified exactly
    # one head per dependent using the pred_heads argument, the scores for the heads
    # specified in heads_pred are the only ones saved. Using the example above,
    # we know that the score at index (1 {second sentence}, 0 {first deprel label}, 1
    # {dependent index}) will be taken from deprels_pred index (1, 0, 1, 2) (12 below).
    # Similarly, (1, 0, 1, 2) will be taken from deprels_pred index (1, 0, 1, 3)
    # (24 below). It helps to work through the example by hand to see that this is
    # correct.
    actual = _deprel_pred_for_heads(deprel_scores_pred, heads_pred)
    # print(actual)
    expected = torch.tensor(
        [
            [[1, 4, 9, 4, 20], [2, 8, 18, 8, 40]],
            [[2, 12, 24, 40, 10], [4, 24, 48, 80, 20]],
            [[3, 30, 9, 36, 60], [6, 60, 18, 72, 120]],
        ]
    )
    assert torch.equal(actual, expected)
