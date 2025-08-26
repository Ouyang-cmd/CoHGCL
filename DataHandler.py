import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from Params import args


class DataHandler:
    def __init__(self,
                 train_matrix: sp.coo_matrix,
                 val_matrix:   sp.coo_matrix,
                 test_matrix:  sp.coo_matrix):
        self.train_matrix  = train_matrix.tocsr()
        self.val_matrix    = val_matrix.tocsr()
        self.test_matrix   = test_matrix.tocsr()

        self.tRNA_count, self.disease_count = self.train_matrix.shape

        self.adj           = None
        self.transpose_adj = None

    def load_data(self):
        adj = (self.train_matrix != 0).astype(np.float32)
        row_sum = np.array(adj.sum(axis=1)).flatten()
        for i in range(adj.shape[0]):
            if row_sum[i] > 0:
                adj.data[adj.indptr[i]:adj.indptr[i + 1]] /= row_sum[i]

        transpose_adj = csr_matrix(adj.transpose())
        col_sum = np.array(transpose_adj.sum(axis=1)).flatten()
        for i in range(transpose_adj.shape[0]):
            if col_sum[i] > 0:
                transpose_adj.data[transpose_adj.indptr[i]:transpose_adj.indptr[i + 1]] /= col_sum[i]

        self.adj           = adj
        self.transpose_adj = transpose_adj

    def get_train_matrix(self):
        return self.train_matrix

    def get_val_matrix(self):
        return self.val_matrix

    def get_test_matrix(self):
        return self.test_matrix
