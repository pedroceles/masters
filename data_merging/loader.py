# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


class DataLoader(object):
    """Classe para ler os dados e trabalhá-los para o conceito de DataMerging"""
    def __init__(self, filename, columns=None, sep=',', norm=False):
        super(DataLoader, self).__init__()
        self.filename = filename
        self.columns = columns
        self.sep = sep
        self.load_data()
        if norm:
            df_norm = (self.data - self.data.mean()) / (self.data.max() - self.data.min())
            df_norm -= df_norm.min()
            self.data = df_norm

    def load_data(self):
        '''Lê um arquivo em .csv'''
        self.data = pd.read_csv(self.filename, sep=self.sep, names=self.columns)

    def hide_data(self, *attrs, **kwargs):
        """apaga alguns valores dos attrs
        Ex: hide_data([1, 2], [3, 4], [5, 6])
        irá dividir os dados em 3 partes
        na primeira parte os atributos [1, 2] serão None, na sugunda os atributos
        [3, 4] serão None e assim por diante

        :attrs: lista com os attrs
        :returns: novo dataFrame com os valores missing

        """
        hidden_partition = kwargs.pop('hidden_partition', None)
        self.hided_data = self.data.copy()
        total_indexes = self.data.shape[0]
        indexes_shuffled = np.arange(total_indexes)
        self.col_indexes_changed = {}
        if not hidden_partition:
            self._hide_dependent(indexes_shuffled, *attrs)
        else:
            self._hide_independent(indexes_shuffled, *attrs, hidden_partition=hidden_partition)
        return self.hided_data

    def _hide_independent(self, indexes_shuffled, *attrs, **kwargs):
        from itertools import chain
        hidden_partition = kwargs['hidden_partition']
        attrs = chain(*attrs)
        total_elements_to_hide = int(len(indexes_shuffled) * hidden_partition)
        for attr in attrs:
            np.random.shuffle(indexes_shuffled)
            elements_to_hide = indexes_shuffled[0:total_elements_to_hide]
            self.hided_data.ix[elements_to_hide, attr] = None
            self.col_indexes_changed[attr] = elements_to_hide

    def _hide_dependent(self, indexes_shuffled, *attrs):
        import math
        split_step = math.floor(indexes_shuffled.shape[0] * 1.0 / len(attrs))
        np.random.shuffle(indexes_shuffled)
        col_indexes_changed = {}
        for i in range(len(attrs) - 1):
            index_to_change = indexes_shuffled[split_step * i:split_step * (1 + i)]
            self.hided_data.ix[index_to_change, attrs[i]] = None
            for j in attrs[i]:
                col_indexes_changed[j] = index_to_change
        i += 1
        index_to_change = indexes_shuffled[split_step * (i):]
        self.hided_data.ix[index_to_change, attrs[i]] = None
        for j in attrs[i]:
            col_indexes_changed[j] = index_to_change
        self.col_indexes_changed = col_indexes_changed

    @staticmethod
    def get_dist_matrix(vectors):
        from scipy.spatial.distance import cdist
        return cdist(vectors, vectors)

    def fill_data(self, backbones, fill_cols=None, n=1):
        '''Preenche os dados faltantes com base na distancia mais próxima dos backbones'''
        if not fill_cols:
            fill_cols = set(range(self.hided_data.shape[1])).difference(backbones)
        fill_cols = set(fill_cols)
        backbone_vectors = self.hided_data.ix[:, backbones]
        backbone_distances = self.get_dist_matrix(backbone_vectors.values)
        # No for, para cada atributo de fil cols, irei ver os atributos nulos
        # para cada atributo preencho os dados com base na media dos n mais
        # distantes
        for col in fill_cols:
            null_indexes = self.hided_data.ix[:, col].isnull()
            if not null_indexes.any():
                continue
            backbones_dists_to_work = backbone_distances[(~null_indexes).values, :]  # pego as distâncias dos vetores backbones onde o valor do atributo não é nulo
            backbone_vectors_to_work = backbone_vectors.ix[~null_indexes]
            index_to_fill = self.hided_data.ix[null_indexes, :].index
            for i in index_to_fill:
                small_dist_indexes = backbones_dists_to_work[:, col].argsort()
                indexes_neighbours = backbone_vectors_to_work.iloc[small_dist_indexes[0:n], :].index
                neighbors_vectors = self.hided_data.ix[indexes_neighbours, :]
                mid_vector = neighbors_vectors.mean(axis=0).values
                self.hided_data.ix[i, col] = mid_vector[col]
