# -*- coding: utf-8 -*-
from data_merging import study, loader
import numpy as np


class MatrixSingleStudy(study.DataSingleStudy):
    LOADER_CLASS = loader.MissingLoader

    def __init__(self, *args, **kwargs):
        n = kwargs.pop('n', 3)
        super(MatrixSingleStudy, self).__init__(*args, **kwargs)
        self.n = n

    def hide_data(self, *attrs):
        self.loader.hide_data(*attrs)
        # self.loader.sort_by_nans()

    @staticmethod
    def fill_partial_data(attr_to_fill, backbones, vectors_to_fill, vectors_to_sort, n=3):
        from scipy.spatial.distance import cdist
        backbones_to_sort = vectors_to_sort[:, backbones]
        backbones_to_fill = vectors_to_fill[:, backbones]
        dist_matrix = cdist(backbones_to_sort, backbones_to_fill)
        for i, vector in enumerate(vectors_to_fill):
            dists = dist_matrix[:, i]
            idx = dists.argsort()[:n]
            value = vectors_to_sort[idx, attr_to_fill].mean()
            vector[attr_to_fill] = value
            vectors_to_fill[i] = vector

        return vectors_to_fill

    @staticmethod
    def get_not_null_vectors(attrs, df):
        """seleciona os vetores do DataFrame df em que nenhum dos atributos em attrs
        é nulo.

        :attrs: list/tuple like
        :df: pandas.DataFrame
        :returns: (np.array) os vetores de df onde nenum dos attrs é nulo

        """
        not_null = ~df.iloc[:, attrs].isnull().any(1)
        new_df = df[not_null]
        return new_df.values

    @staticmethod
    def get_null_vectors(attrs, df):
        """Oposto de get_not_null_vectors
        :returns: (np.array) os vetores de df onde algum dos attrs é nulo

        """
        null = df.iloc[:, attrs].isnull().any(1)
        new_df = df[null]
        return new_df.values

    def calc_item_matrix(self, null_attr, null_group, estimator):
        """Calcula o item a(i,j) da matriz. Cada item da matriz representa o erro de
        classifição dos vetores v. Onde v são os vetores onde i é nulo e foi estimado
        utilizado rodos os atributos sem ser i e j

        :i: int
        :j: int
        :estimator: tenha um método fit e predict (scikit-learn estimators)
        :returns: (float) o erro ou None se i==j

        """
        import pandas as pd
        if null_attr in null_group:
            return None
        hided_data = self.loader.hided_data

        vectors_null = self.get_null_vectors([null_attr], hided_data)
        total_attr = range(self.n_attr)
        est_attrs = list(set(total_attr).difference([null_attr] + list(null_group)))
        vectors_estimators = self.get_not_null_vectors(sorted([null_attr] + est_attrs), hided_data)

        vector_filled = self.fill_partial_data(null_attr, est_attrs, vectors_null, vectors_estimators, self.n)
        score = self.calc_score(estimator, pd.DataFrame(vector_filled), [null_attr], bool(self.seed))
        return abs(score.mean())

    def calc_matrix(self, estimator):
        from base import Matrix
        self.groups = self.get_groups(self.loader.hided_data)
        attrs_null = []
        keys = sorted(self.groups.keys())
        for k in keys:
            attrs_null.extend(k)
        attrs_null = sorted(set(attrs_null))
        # matrix = np.empty((len(attrs_null), len(keys)), type(None))
        matrix = Matrix(attrs_null, keys)
        for (i, attr_null) in enumerate(attrs_null):
            for (j, group) in enumerate(keys):
                matrix[i, j] = self.calc_item_matrix(attr_null, group, estimator)
        self.matrix = matrix
        return matrix

    def calc_best_groups(self):
        idx = np.nanargmin(self.matrix.values, axis=1)
        return self.matrix.cols[idx]

    def fill_data(self):
        total_attr = set(range(self.n_attr))
        best_groups = self.calc_best_groups()
        self.filled_data = self.loader.hided_data.copy()
        values = self.filled_data.values
        for attr, group in zip(self.matrix.rows, best_groups):
            backbones = list(total_attr.difference(group))
            backbones_vectors = self.get_not_null_vectors(sorted([attr] + list(backbones)), self.loader.hided_data)
            vectors_to_fill = self.get_null_vectors([attr], self.loader.hided_data)
            vectors_filled = self.fill_partial_data(attr, backbones, vectors_to_fill, backbones_vectors, self.n)
            null = self.filled_data.iloc[:, attr].isnull()
            # self.filled_data.iloc[null.values].iloc[:, attr] = vectors_filled[:, attr]
            values[null.values, attr] = vectors_filled[:, attr]
        self.filled_data[:] = values

    @staticmethod
    def get_groups(df):
        '''Pega os grupos que tem dado vazio'''
        from collections import defaultdict
        isnan = df.isnull().values
        asort = isnan.argsort()
        sums = isnan.sum(axis=1)
        results = defaultdict(list)
        for idx, (vec, size) in enumerate(zip(asort, sums)):
            value = ()
            if size > 0:
                value = tuple(vec[-size:])
            results[value].append(idx)
        return results
