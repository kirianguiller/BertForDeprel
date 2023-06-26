from BertForDeprel.parser.utils.chuliu_edmonds_utils import chuliu_edmonds, chuliu_edmonds_one_root, chuliu_edmonds_one_root_with_constrains
import numpy as np


mock_coefs_lists = [
    [-9.9, -6.6, -1.2, -6.5, -8.7, -8.3],
    [-14.9, -13.2, 0.5, -13, -9.4, -7.9],
    [1.6, -3.8, -7.4, -3.9, -14.9, -20.1],
    [-1.8, -4.3, -0.9, -2.6, -6.1, -7.7],
    [-9.7, -8.5, -8.7, -1.0, -5.7, -8.8],
    [-7.9, -4.9, -4.5, -1.2, -3.0, -4.1]
                     ]

mock_coefs_array = np.array(mock_coefs_lists).astype(np.float64)

def test_chuliu_edmonds_one_root():
    assert chuliu_edmonds_one_root(mock_coefs_array).tolist() == [0, 2, 0, 2, 3, 3]

def test_chuliu_edmonds_one_root_with_constrains():
    assert chuliu_edmonds_one_root_with_constrains(mock_coefs_array).tolist() == [0, 2, 0, 2, 3, 3]
    assert chuliu_edmonds_one_root_with_constrains(mock_coefs_array, []).tolist() == [0, 2, 0, 2, 3, 3]
    assert chuliu_edmonds_one_root_with_constrains(mock_coefs_array, [(4, 5)]).tolist() == [0, 2, 0, 2, 5, 3]
    assert chuliu_edmonds_one_root_with_constrains(mock_coefs_array, [(1, 0)]).tolist() == [0, 0, 1, 2, 3, 3]
    assert chuliu_edmonds_one_root_with_constrains(mock_coefs_array, [(3, 0)]).tolist() == [0, 2, 3, 0, 3, 3]
    assert chuliu_edmonds_one_root_with_constrains(mock_coefs_array, [(4, 1), (4, 5)]).tolist() == [0, 2, 0, 2, 1, 3]
    assert chuliu_edmonds_one_root_with_constrains(mock_coefs_array, [(5, 1), (5, 4)]).tolist() == [0, 2, 0, 2, 3, 4]
