# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from matrix import study


class ImputationStudy(study.MatrixSingleStudy):

    def __init__(self, *args, **kwargs):
        super(ImputationStudy, self).__init__(*args, **kwargs)
        self.groups = {}

    def fill_single_data(self, vector_to_fill, attr_to_fill):
        '''Fill the nil value of `attr_to_fill` in `vector_to_fill`, using all non-nil
        values of vector_to_fill as backbones'''

        attrs_filled = np.argwhere(~np.isnan(vector_to_fill[:-1].astype(float)))
        attrs_filled = attrs_filled.reshape(attrs_filled.shape[0])
        vectors_to_sort = self.get_not_null_vectors(sorted(list(attrs_filled) + [attr_to_fill]), self.loader.hided_data)
        if vectors_to_sort.shape[0] == 0:
            group = self.groups.get(attr_to_fill)
            if not group:
                groups = self.get_not_null_groups(attr_to_fill)
                group = max(groups.items(), key=lambda x: x[1])[0]
                self.groups[attr_to_fill] = group
            vectors_to_sort = self.get_not_null_vectors(sorted(group), self.loader.hided_data)
            attrs_filled = np.array(group)

        vector_filled = self.fill_partial_data(attr_to_fill, attrs_filled, np.array([vector_to_fill]), vectors_to_sort, self.n)[0]
        return vector_filled

    def fill_data(self):
        from copy import copy
        self.filled_data = self.loader.hided_data.copy()
        filled_values = self.filled_data.values
        for index, vector in enumerate(filled_values):
            v = copy(vector)
            attrs_null = np.argwhere(np.isnan(v[:-1].astype(float)))
            attrs_null = attrs_null.reshape(attrs_null.shape[0])
            if not attrs_null.shape[0]:
                continue
            for attr_null in attrs_null:
                v = self.fill_single_data(v, attr_null)
            filled_values[index] = v

        self.filled_data.iloc[:] = filled_values

    def get_not_null_groups(self, attr):
        '''Pega todos os grupos em que attr NÃO é null'''
        from collections import defaultdict
        df = self.loader.hided_data.iloc[:, :-1]
        not_null_df = ~df.iloc[:, attr].isnull()
        dfn = df[not_null_df]
        not_null_dfn = ~dfn.isnull()
        values = not_null_dfn.values
        asort = values.argsort()
        sums = values.sum(axis=1)
        results = defaultdict(int)
        for size, vec in zip(sums, asort):
            vector = tuple(vec[-size:])
            results[vector] += 1
        return results

    def compare(self, estimator, **calc_score_kwargs):
        '''compara a previsão usando filled_data contra full_data'''
        full_score = abs(self.calc_score(estimator, self.loader.full_data, **calc_score_kwargs).mean())
        filled_score = abs(self.calc_score(estimator, self.filled_data, **calc_score_kwargs).mean())
        return full_score, filled_score
