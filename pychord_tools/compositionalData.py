import numpy as np
from sklearn import preprocessing


def substituteZeros(chromas):
    data = np.copy(chromas)
    data[data == 0] = 1e-08
    return data

def amalgamate(partitions, chromas):
    res = np.empty((len(partitions), chromas.shape[0]), chromas.dtype)
    for i in range(len(partitions)):
        res[i, :] =  np.sum(chromas[:, partitions[i]], axis=1)
    return res.transpose()

def subcomposition(partitions, chromas):
    res = np.empty((len(partitions), chromas.shape[0]), chromas.dtype)
    for i in range(len(partitions)):
        res[i] =  np.sum(chromas[:, partitions[i]], axis=1)
    return preprocessing.normalize(res.transpose(), norm='l1')

def alr(chromaVector) :
    # alr to tonic.
    return np.log(chromaVector[1:len(chromaVector)] / chromaVector[0])

def alrinv(vector) :
    # alr to tonic.
    return preprocessing.normalize([np.concatenate(([1], np.exp(vector)))], norm='l1')[0]
