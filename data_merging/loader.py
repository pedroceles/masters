# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


class DataLoader(object):
    """Classe para ler os dados e trabalhá-los para o conceito de DataMerging"""
    def __init__(self, filename, columns=None, sep=',', norm=False, test_split=0.2):
        super(DataLoader, self).__init__()
        self.filename = filename
        self.columns = columns
        self.sep = sep
        self.load_data()
        if norm:
            df_norm = self.full_data.iloc[:, :-1]
            df_norm = (df_norm - df_norm.mean()) / (df_norm.max() - df_norm.min())
            df_norm -= df_norm.min()
            self.full_data.iloc[:, :-1] = df_norm.values
        if test_split:
            from sklearn.cross_validation import ShuffleSplit
            train, test = next(iter(ShuffleSplit(self.full_data.shape[0], 1, test_split)))
            self.data = self.full_data.iloc[train].copy()
            self.test_data = self.full_data.iloc[test].copy()
        else:
            self.data = self.full_data.copy()
            self.test_data = pd.DataFrame([])

    def load_data(self):
        '''Lê um arquivo em .csv'''
        self.full_data = pd.read_csv(self.filename, sep=self.sep, names=self.columns)

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
            self.hided_data.iloc[elements_to_hide, attr] = None
            self.col_indexes_changed[attr] = elements_to_hide

    def _hide_dependent(self, indexes_shuffled, *attrs):
        import math
        split_step = math.floor(indexes_shuffled.shape[0] * 1.0 / len(attrs))
        np.random.shuffle(indexes_shuffled)
        col_indexes_changed = {}
        for i in range(len(attrs) - 1):
            index_to_change = indexes_shuffled[split_step * i:split_step * (1 + i)]
            self.hided_data.iloc[index_to_change, attrs[i]] = None
            for j in attrs[i]:
                col_indexes_changed[j] = index_to_change
        i += 1
        index_to_change = indexes_shuffled[split_step * (i):]
        self.hided_data.iloc[index_to_change, attrs[i]] = None
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


class MissingLoader(DataLoader):
    def hide_data(self, *attrs):
        '''Esconde os dados de forma que só haja uma coluna em branco em cada linha
        eque haja sempre alguma coluna em branco em todas as linhas.
        Se sort for True, ordena pelas colunas que estão em branco.'''
        self.hided_data = self.data.copy()
        size = self.data.shape[0]
        idx = np.arange(size)
        np.random.shuffle(idx)
        number_column_nas = size / len(attrs)
        for i, attr in enumerate(attrs):
            if i == len(attrs) - 1:
                _idx = idx[number_column_nas * i:]
            else:
                _idx = idx[number_column_nas * i: number_column_nas * (i + 1)]
            self.hided_data.iloc[_idx, attr] = None

    def hide_data_independent(self, *attrs, **kwargs):
        self.hided_data = self.data.copy()
        percent = kwargs.pop('percent', 0.5)
        size = self.data.shape[0]
        n_items = int(size * percent)
        idx = np.arange(size)
        for attr in attrs:
            np.random.shuffle(idx)
            self.hided_data.iloc[idx[:n_items], attr] = None

    def hide_extra(self, *attrs, **kwargs):
        percent = kwargs['percent']
        shape = self.hided_data.iloc[:, attrs].shape
        total_items = shape[0] * shape[1]
        null_items = self.hided_data.isnull()
        total_null = null_items.sum().sum()
        now_percent = total_null / float(total_items)
        assert percent > now_percent, "new percent must be bigger"

        extra_to_hide_by_attr = int(shape[0] * (percent - now_percent))
        for attr in attrs:
            null_attr = null_items.iloc[:, attr]
            df_attr = self.hided_data.iloc[:, attr]
            not_null_indexes = df_attr[~null_attr].index
            indexes_to_null = np.random.choice(not_null_indexes, extra_to_hide_by_attr, False)
            self.hided_data.iloc[indexes_to_null, attr] = None

    def sort_by_nans(self):
        """Ordena pelo índice da coluna que tem dado faltante
        :returns: None

        """
        values = self.hided_data.values
        nans = np.isnan(values)
        idx = nans.argsort()[:, -1].argsort()
        values = values[idx, :]
        self.hided_data.iloc[:] = values
