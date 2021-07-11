import numpy as np
import pandas as pd



# Precision at K
def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:,:k] != 0
    if r.shape[1] < k:
        raise ValueError('Relevance score length < k')
    r  = np.sum(r, axis=1)
    return np.sum(r>0)/r.shape[0]


# Mean Reciprocal Rank
def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

resnet1000 = np.load('data\\model_inference\\shopping100k\\resnet50_1000\\ckpt5\\evaluate70.npy')[:, 1:]
resnet2000 = np.load('data\model_inference\shopping100k\Multinet\multinet_resnet50_2000\ckpt4\evaluate70.npy')[:, 1:]
densenet128 = np.load('data\\model_inference\\shopping100k\\result_emb128\\evaluate70.npy')[:, 1:]
print(resnet1000.shape)

precision_K = []
reciprocal = []
for k in range(71):
    if k == 0:
        precision_K.append([0, 0, 0, 0])
        reciprocal.append([0, 0, 0, 0])
    else:
        precision_K.append([k, precision_at_k(resnet1000[:, :k], k), precision_at_k(resnet2000[:, :k], k), precision_at_k(densenet128[:, :k], k)])
        reciprocal.append([k, mean_reciprocal_rank(resnet1000[:, :k]), mean_reciprocal_rank(resnet2000[:, :k]), mean_reciprocal_rank(densenet128[:, :k])])
pd.DataFrame(precision_K, columns=['k', 'resnet50_1000', 'resnet50_2000', 'densenet161_128']).to_csv('fashion-visual-search\src\EDA\metric\precision_at_K.csv', index=False)
pd.DataFrame(reciprocal, columns=['k', 'resnet50_1000', 'resnet50_2000', 'densenet161_128']).to_csv('fashion-visual-search\src\EDA\metric\mean_reciprocal_rank.csv', index=False)
