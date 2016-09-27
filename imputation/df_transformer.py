# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class BaseDFTransformer(object):

    def __init__(self, f_path=None, db_name=None, data=None):
        self.f_path = f_path
        self.db_name = db_name
        self.data = data

    def open_file(self):
        return open(self.f_path)

    def get_pickle(self):
        import pickle
        return self.data or pickle.load(self.open_file())


class PercentDFTransformer(BaseDFTransformer):

    def __init__(self, *args, **kwargs):
        self.div = kwargs.pop('div', True)
        super(PercentDFTransformer, self).__init__(*args, **kwargs)

    def treat_data(self, array):
        base = array[:, 1:].T
        if self.div:
            base = np.array([array[:, i] / array[:, 0] for i in range(1, 5)])
        mean = np.mean(base, 1)
        p25 = np.percentile(base, 25, 1)
        p75 = np.percentile(base, 75, 1)
        return np.array([mean, p25, p75])

    def get_df(self):
        gen_info_cols = ['db', 'estimator', 'p_attr', 'p_row']
        data_cols = [
            ['biased', 'random'],
            ['neigh', 'naive', 'no_rows', 'no_cols'],
            ['mean', 'p25', 'p75']
        ]
        data_col_mi = pd.MultiIndex.from_product(data_cols)
        df_data = pd.DataFrame(columns=data_col_mi)
        df_info = pd.DataFrame(columns=gen_info_cols)
        dict_ = self.get_pickle()
        for index, ((estimator, p_attr, p_row), data) in enumerate(dict_.items()):
            biased = self.treat_data(data['biased']).T
            random = self.treat_data(data['random']).T
            row_array = np.array([biased, random]).flatten()

            df_info.loc[index] = [self.db_name, estimator, p_attr, p_row]
            df_data.loc[index] = row_array

        df = pd.concat([df_info, df_data], axis=1)
        df.set_index(['db', 'estimator', 'p_attr', 'p_row'], inplace=True)
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

    def __init__(self, df_transformer_klass, files=None, data=None, names=None, **kwargs):
        self.df_transformer_klass = df_transformer_klass
        self.files = files
        self.names = names
        self.df_transformer_klass_kwargs = kwargs
        self.data = data
        if not self.names:
            if data:
                self.name = data.keys()
            else:
                self.names = [n.split('/')[-3] for n in self.files]

    def get_df(self):
        dfs = []
        if self.files:
            for f_path, name in zip(self.files, self.names):
                transformer = self.df_transformer_klass(
                    f_path, name, **self.df_transformer_klass_kwargs)
                dfs.append(transformer.get_df())
        elif self.data:
            for name, value in self.data.items():
                transformer = self.df_transformer_klass(
                    None, name, data=value, **self.df_transformer_klass_kwargs)
                dfs.append(transformer.get_df())
        df = pd.concat(dfs, axis=0)
        df.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)
        return df


class PickleAggregator(object):
    def __init__(self, pattern):
        from glob import glob
        import pickle
        fs = glob(pattern)
        aggregate = {}
        for f in fs:
            data = pickle.load(open(f))
            db = f.split('/')[-3]
            aggregate[db] = data
        self.aggregate = aggregate


def get_agg_pickle_df(fpath, **kwargs):
    kwargs.setdefault('df_transformer_klass', PercentDFTransformer)
    import pickle
    data = pickle.load(open(fpath))
    kwargs['data'] = data
    df = DFAggregator(**kwargs).get_df()
    return df
