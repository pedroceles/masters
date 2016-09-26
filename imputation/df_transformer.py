# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class BaseDFTransformer(object):

    def __init__(self, f_path, db_name):
        self.f_path = f_path
        self.db_name = db_name

    def open_file(self):
        return open(self.f_path)

    def get_pickle(self):
        import pickle
        return pickle.load(self.open_file())


class PercentDFTransformer(BaseDFTransformer):

    def __init__(self, *args, **kwargs):
        self.div = kwargs.pop('div', True)
        super(PercentDFTransformer, self).__init__(*args, **kwargs)

    def treat_data(self, array):
        base = array[:, 1:].T
        if self.div:
            base = np.array([array[:, i] / array[:, 0] for i in range(1, 4)])
        mean = np.mean(base, 1)
        p25 = np.percentile(base, 25, 1)
        p75 = np.percentile(base, 75, 1)
        return np.array([mean, p25, p75])

    def get_df(self):
        gen_info_cols = ['db', 'estimator', 'percent']
        data_cols = [
            ['biased', 'random'],
            ['neigh', 'naive', 'removed'],
            ['mean', 'p25', 'p75']
        ]
        data_col_mi = pd.MultiIndex.from_product(data_cols)
        df_data = pd.DataFrame(columns=data_col_mi)
        df_info = pd.DataFrame(columns=gen_info_cols)
        dict_ = self.get_pickle()
        for index, ((estimator, percent), data) in enumerate(dict_.items()):
            biased = self.treat_data(data['biased']).T
            random = self.treat_data(data['random']).T
            row_array = np.array([biased, random]).flatten()

            df_info.loc[index] = [self.db_name, estimator, percent]
            df_data.loc[index] = row_array

        df = pd.concat([df_info, df_data], axis=1)
        df.set_index(['db', 'estimator', 'percent'], inplace=True)
        multi_index = pd.MultiIndex.from_tuples(df.columns)
        df.columns = multi_index
        df.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)
        return df


class AbsoluteDFTransformer(BaseDFTransformer):
    def get_df(self):
        gen_info_cols = ['db', 'estimator', 'percent', 'error']
        df_data = pd.DataFrame(columns=gen_info_cols)
        dict_ = self.get_pickle()
        for index, ((estimator, percent), data) in enumerate(dict_.items()):
            complete = data['biased'].mean(0)
            complete = complete[0]
            df_data.loc[index] = [self.db_name, estimator, percent, complete]
        df = df_data.set_index(['db', 'estimator', 'percent'])
        df.sort_index(inplace=True)
        return df.groupby(level=['db', 'estimator']).mean()


class DFAggregator(object):

    """Docstring for DFAggregator. """

    def __init__(self, df_transformer_klass, files, names=None, **kwargs):
        self.df_transformer_klass = df_transformer_klass
        self.files = files
        self.names = names
        self.df_transformer_klass_kwargs = kwargs
        if not self.names:
            self.names = [n.split('/')[-3] for n in self.files]

    def get_df(self):
        dfs = []
        for f_path, name in zip(self.files, self.names):
            transformer = self.df_transformer_klass(
                f_path, name, **self.df_transformer_klass_kwargs)
            dfs.append(transformer.get_df())
        df = pd.concat(dfs, axis=0)
        df.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)
        return df
