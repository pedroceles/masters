# -*- coding: utf-8 -*-
import tempfile
import unittest

import pandas
import numpy as np

from data_merging.loader import DataLoader


class WriteFileMixin(object):
    def write_file(self):
        self.data_file = tempfile.TemporaryFile()
        self.data_file.write('A,B,C,Class\n')
        self.data_file.write('0,0,0,0\n')
        self.data_file.write('1,1,1,0\n')
        self.data_file.write('0,1,1,0\n')
        self.data_file.write('0,0,1,0\n')
        self.data_file.write('1,1,1,0\n')
        self.data_file.write('1,1,0,0\n')
        self.data_file.write('1,1,0,0\n')
        self.data_file.seek(0)
        return self.data_file


class TestDataLoader(unittest.TestCase):

    """Test the data Loader Class"""

    def setUp(self):
        data_file = tempfile.TemporaryFile()
        data_file.write('A,B,C,Class\n')
        data_file.write('0,0,0,0\n')
        data_file.write('1,1,1,0\n')
        data_file.write('0,1,1,0\n')
        data_file.write('0,0,1,0\n')
        data_file.write('1,1,1,0\n')
        data_file.write('1,1,0,0\n')
        data_file.write('1,1,0,0\n')
        data_file.seek(0)
        self.loader = DataLoader(data_file, test_split=0)

    def tearDown(self):
        pass

    def test_setUp(self):
        pass

    def test_hide_data(self):
        hided_data = self.loader.hide_data([1], [2])
        self.assertEqual(7, hided_data.isnull().sum().sum(), 'Null points=7')
        self.assertEqual(3, hided_data.ix[:, 1].isnull().sum())
        self.assertEqual(4, hided_data.ix[:, 2].isnull().sum())

        self.setUp()
        hided_data = self.loader.hide_data([0], [1], [2])
        self.assertEqual(2, hided_data.ix[:, 0].isnull().sum())
        self.assertEqual(2, hided_data.ix[:, 1].isnull().sum())
        self.assertEqual(3, hided_data.ix[:, 2].isnull().sum())

        self.setUp()
        hided_data = self.loader.hide_data([0], [1], [2], hidden_partition=0.7)
        self.assertEqual(4, hided_data.ix[:, 0].isnull().sum())
        self.assertEqual(4, hided_data.ix[:, 1].isnull().sum())
        self.assertEqual(4, hided_data.ix[:, 2].isnull().sum())

    def test_fill_data(self):
        hided_data = self.loader.data.copy()
        hided_data.ix[:2, 0] = None
        self.loader.hided_data = hided_data
        self.loader.fill_data([1, 2])
        self.assertFalse(pandas.isnull(self.loader.hided_data.ix[0, 0]))
        self.assertFalse(pandas.isnull(self.loader.hided_data.ix[1, 0]))

        def _new_df():
            data_file = tempfile.TemporaryFile()
            data_file.write('A,B,C,Class\n')
            data_file.write('0,0,0,0\n')
            data_file.write('1,1,1,0\n')
            data_file.write('0.1,0,1,0\n')
            data_file.write('0.2,1,0,0\n')
            data_file.seek(0)
            self.loader = DataLoader(data_file)

        # reescrevendo para ver se pega a média exata
        _new_df()
        hided_data = self.loader.data.copy()
        hided_data.ix[0, 0] = None
        hided_data.ix[3, 3] = None
        self.loader.hided_data = hided_data
        self.loader.fill_data([1, 2], n=2)
        self.assertAlmostEqual(0.15, self.loader.hided_data.ix[0, 0], 5)
        self.assertEqual(0, self.loader.hided_data.ix[3, 3])

        # Verificando se todos os outros dados estão iguais
        for i in range(4):
            for j in range(4):
                if [i, j] not in [[0, 0], [3, 3]]:
                    self.assertEqual(self.loader.data.ix[i, j], self.loader.hided_data.ix[i, j])

    def test_get_dist_matrix(self):
        import math
        matrix = self.loader.get_dist_matrix([[0, 0, 0], [1, 0, 0], [1, 1, 1]])
        for i in range(3):
            self.assertEqual(0, matrix[i, i])
        self.assertEqual(1, matrix[0, 1])
        self.assertAlmostEqual(math.sqrt(3), matrix[0, 2], 5)
        self.assertAlmostEqual(math.sqrt(2), matrix[1, 2], 5)


class TestMissingLoader(WriteFileMixin, unittest.TestCase):

    def setUp(self):
        from data_merging.loader import MissingLoader
        data_file = self.write_file()
        self.loader = MissingLoader(data_file, test_split=0)

    def tearDown(self):
        pass

    def test_hide_data(self):
        import mock
        old_shuffle = np.random.shuffle
        np.random.shuffle = mock.Mock(side_effect=lambda x: None)
        self.loader.hide_data(0, 1, 2)
        self.assertEqual(2, self.loader.hided_data.iloc[0:2, 0].isnull().sum())
        self.assertEqual(2, self.loader.hided_data.iloc[2:4, 1].isnull().sum())
        self.assertEqual(3, self.loader.hided_data.iloc[4:7, 2].isnull().sum())
        np.random.shuffle = old_shuffle

    def test_sort_by_nans(self):
        self.loader.hide_data(0, 1, 2)
        self.loader.sort_by_nans()
        self.assertEqual(2, self.loader.hided_data.iloc[0:2, 0].isnull().sum())
        self.assertEqual(2, self.loader.hided_data.iloc[2:4, 1].isnull().sum())
        self.assertEqual(3, self.loader.hided_data.iloc[4:7, 2].isnull().sum())
        self.assertEqual(7, self.loader.hided_data.isnull().sum().sum())
