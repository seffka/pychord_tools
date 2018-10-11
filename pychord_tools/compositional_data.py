import numpy as np
from sklearn import preprocessing


def substitute_zeros(data, copy=True):
    if copy:
        data = np.copy(data)
    data[data < 1e-08] = 1e-08
    return data


def amalgamate(partitions, chromas):
    res = np.empty((len(partitions), chromas.shape[0]), chromas.dtype)
    for i in range(len(partitions)):
        res[i, :] = np.sum(chromas[:, partitions[i]], axis=1)
    return res.transpose()


def subcomposition(partitions, chromas):
    res = np.empty((len(partitions), chromas.shape[0]), chromas.dtype)
    for i in range(len(partitions)):
        res[i] = np.sum(chromas[:, partitions[i]], axis=1)
    return preprocessing.normalize(substitute_zeros(res.transpose()), norm='l1')


def alr(chroma_vector):
    # alr to tonic.
    return np.log(chroma_vector[1:len(chroma_vector)] / chroma_vector[0])


def alrinv(vector):
    # inverse alr to tonic.
    return preprocessing.normalize([np.concatenate(([1], np.exp(vector)))], norm='l1')[0]
