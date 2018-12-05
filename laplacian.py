
import numpy as np
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import csc_matrix
from scipy.sparse import diags
from scipy.sparse import lil_matrix
import scipy.sparse as sp

import numpy as np
from scipy import sparse

from time import time


def sparsemax(X, Y):
    # the indices of all non-zero elements in both arrays
    idx = np.hstack((X.nonzero(), Y.nonzero()))

    # find the set of unique non-zero indices
    idx = tuple(unique_rows(idx.T).T)

    # take the element-wise max over only these indices
    X[idx] = np.maximum(X[idx].A, Y[idx].A)

    return X


def unique_rows(a):
    void_type = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    b = np.ascontiguousarray(a).view(void_type)
    idx = np.unique(b, return_index=True)[1]
    return a[idx]


def load_embedding(filename, headings):
    file = open(filename)
    headings = [int(x) for x in file.readline().split()]
    nums = headings[0]
    dim = headings[1]
    embed = np.zeros((nums, dim))
    lines = file.readlines()
    file.close()
    for l in lines:
        s = [float(x) for x in l.split()]
        embed[int(s[0])-1] = s[1:]
    return embed


def edgelist_to_adjacency(edgelist, heading):
    adj_matrix = np.zeros((heading[0], heading[0]))

    for edge in edgelist:
        if len(edge)>=3:
            adj_matrix[edge[0] - 1, edge[1] - 1] = edge[2]
            adj_matrix[edge[1] - 1, edge[0] - 1] = edge[2]
        else:
            adj_matrix[edge[0] - 1, edge[1] - 1] = 1
            adj_matrix[edge[1] - 1, edge[0] - 1] = 1

    return adj_matrix


def write_embedding(embed, heading, filename):
    f = open(filename, 'w+')
    heading_str = [str(x) + ' ' for x in heading]
    f.write(''.join(heading_str) + '\n')
    for line in embed:
        text = [str(x) + ' ' for x in line]
        text.append('\n')
        f.write(''.join(text))

    f.close()


def load_edgelist(filename):
    file = open(filename)
    edges = [[int(x) for x in l.split()] for l in file.readlines()]
    file.close()
    heading = edges[0]
    edges = np.array(edges[1:])

    return heading, edges


def edgelist_to_adjacency_sparse(edgelist, heading):
    print('transforming to adjacency matrix(sparse)...')
    adj_matrix = lil_matrix((heading[0], heading[0]))
    for edge in edgelist:
        if len(edge) >= 3:
            adj_matrix[edge[0] - 1, edge[1] - 1] = edge[2]
            adj_matrix[edge[1] - 1, edge[0] - 1] = edge[2]
        else:
            adj_matrix[edge[0] - 1, edge[1] - 1] = 1
            adj_matrix[edge[1] - 1, edge[0] - 1] = 1

    return adj_matrix.tocsc()


def laplacian_eigenmap_sparse(W, n, scale=0.5):
    degrees = W.sum(axis=1)
   # matrix_size = W.shape[0] * W.shape[1]
   # print((W!=0).sum())
   # print(W.todense())
    degrees = np.asarray(degrees).flatten()
    max_degree = np.max(degrees)

    if scale!=0:
        comm = W.dot(W.T) 
        W = comm - diags(comm.diagonal(), 0) + W
        
        scale_factor = np.power( (1 / degrees), scale)
        scale_matrix = diags(scale_factor, 0)
        W = scale_matrix.dot(W).dot(scale_matrix)

    print('computing ld...')
    # this is to normalize the weight matrix.
    ld = diags(np.power(np.asarray(W.sum(axis=1)).flatten(), -0.5), 0)
    print('computing D0...')
    D0 = ld.dot(W).dot(ld)
    print('computing max(D0, D0.T)...')
    D0 = sparsemax(D0, D0.T)
    print('carrying eigenvalue decomposition...')
    start = time()
    values, V = eigsh(D0, n, which='LA')
    print('time consumed: %0.2f seconds' % (time()-start))
    indexes = np.arange(1, len(degrees) + 1)
    indexes = indexes.reshape((len(indexes), 1))
    V = np.hstack((indexes, V))
    return V, W

