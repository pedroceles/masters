# -*- coding: utf-8 -*-

import pandas as pd


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

    def get_df(self):
        from itertools import chain
        gen_info_cols = ['db', 'estimator', 'percent']
        value_info_cols = ['complete', 'neigh', 'naive', 'removed']
        lvl_1_cols = value_info_cols * 2
        lvl_0_cols = (
            ['biased'] * len(value_info_cols)
            + ['random'] * len(value_info_cols)
        )
        df_data = pd.DataFrame(columns=[lvl_0_cols, lvl_1_cols])
        df_info = pd.DataFrame(columns=gen_info_cols)
        dict_ = self.get_pickle()
        for index, ((estimator, percent), data) in enumerate(dict_.items()):
            biased = data['biased']
            random = data['random']

            biased_mean = biased.mean(axis=0)
            random_mean = random.mean(axis=0)
            df_info.loc[index] = [self.db_name, estimator, percent]
            df_data.loc[index] = list(chain(biased_mean, random_mean))

        df = pd.concat([df_info, df_data], axis=1)
        df.set_index(['db', 'estimator', 'percent'], inplace=True)
        multi_index = pd.MultiIndex.from_tuples(df.columns)
        df.columns = multi_index
        df.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)
        return df


class DFAggregator(object):

    """Docstring for DFAggregator. """

    def __init__(self, df_transformer_klass, files, names=None):
        self.df_transformer_klass = df_transformer_klass
        self.files = files
        self.names = names
        if not self.names:
            self.names = [n.split('/')[-3] for n in self.files]

    def get_df(self):
        dfs = []
        for f_path, name in zip(self.files, self.names):
            transformer = self.df_transformer_klass(f_path, name)
            dfs.append(transformer.get_df())
        df = pd.concat(dfs, axis=0)
        df.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)
        return df
