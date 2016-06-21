# -*- coding: utf-8 -*-
import pandas as pd


class BaseTreatment(object):

    """Class to serve as model for data treatment"""

    def __init__(self, source_path, save_file_path=None, categorical=True):
        """TODO: to be defined1.

        :source_path: TODO
        :save_file_path: TODO
        :categorical: TODO

        """
        self._source_path = source_path
        self._save_file_path = save_file_path
        self._categorical = categorical

    def run(self):
        df = self.read_file()
        values = df.values
        processed_data = self.preprocess(values)
        processed_df = pd.DataFrame(processed_data)
        self.write_file(processed_df)

    def get_destination_file(self):
        f = self._save_file_path
        if not f:
            import inspect
            import os
            dir_ = os.path.dirname(inspect.getfile(self.__class__))
            return os.path.join(dir_, '../', 'treated.csv')
        return f

    def read_file(self, *args, **kwargs):
        '''Read csv file and sets self.df, can be overriden'''
        header = kwargs.pop('header', None)
        self._df = pd.read_csv(self._source_path, *args, header=header, **kwargs)
        return self._df

    def write_file(self, df):
        df.to_csv(self.get_destination_file(), header=None, index=False)

    def preprocess(self, values):
        '''Preprocess the values, should return the matrix representing the values, np.array'''
        return values
