# -*- coding: utf-8 -*-
import unittest
import mock

import numpy as np
import pandas as pd

from data_merging.test import data_loader
from matrix.study import MatrixSingleStudy


class TestMatrix(data_loader.WriteFileMixin, unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        self.write_file()
        self.mss = MatrixSingleStudy(self.data_file, test_split=0)

    def test_setUp(self):
        from data_merging import loader
        self.assertIsInstance(self.mss.loader, loader.MissingLoader)
        self.assertEqual((7, 4), self.mss.loader.data.shape)

    def test_hide_data(self):
        self.mss.loader.hide_data = mock.Mock()
        self.mss.loader.sort_by_nans = mock.Mock()
        self.mss.hide_data(*[1, 2])
        self.mss.loader.hide_data.assert_called_with(*[1, 2])
        self.mss.loader.sort_by_nans.assert_called_with()

    def test_fill_partial_data(self):
        vectors_to_fill = np.array([
            [None, 1, 1, 1, 1],
            [None, 0, 0, 0, 0],
            [None, 1, 0, 1, 0],
        ]
        )
        vectors_to_sort = np.array([
            [1, 1, 0, 1.1, None],  # selecionado pelo 3o
            [0.5, 1.1, 0, 1, None],  # selecionado pelo 3o
            [2, 0.1, 0.1, 0, None],  # selecionado pelo 2o
            [3, 0.1, 0.2, 0, None],  # selecionado pelo 2o
            [0.5, 1, 0.8, 0.9, None],  # selecionado pelo 1o
            [0, 1, 1, 1.1, None],  # selecionado pelo 1o
        ])
        filled_data = self.mss.fill_partial_data(attr_to_fill=0, backbones=[2, 3], vectors_to_fill=vectors_to_fill, vectors_to_sort=vectors_to_sort, n=2)
        self.assertEqual((3, 5), filled_data.shape)
        self.assertAlmostEqual(np.mean([0, 0.5]), filled_data[0, 0])
        self.assertAlmostEqual(np.mean([2, 3]), filled_data[1, 0])
        self.assertAlmostEqual(np.mean([1, 0.5]), filled_data[2, 0])

    def get_big_df(self):
        a = np.array([
            [0,     1, 2,    3, 4,    5,    6,    7,    8,    9, 10],
            [1,     1, 1, None, 1,    1,    1,    1,    1,    1,  1],
            [2,     1, 1, None, 1,    1,    1,    1,    1,    1,  1],
            [3,  None, 1, None, 1, None,    1,    1,    1,    1,  1],
            [4,  None, 1, None, 1, None,    1,    1,    1,    1,  1],
            [5,  None, 1,    1, 1,    1, None,    1,    1,    1,  1],
            [6,  None, 1,    1, 1,    1, None, None,    1,    1,  1],
            [7,  None, 1,    1, 1,    1, None, None,    1,    1,  1],
            [8,  None, 1,    1, 1,    1,    1,    1, None,    1,  1],
            [9,  None, 1,    1, 1,    1,    1,    1, None,    1,  1],
            [10, None, 1,    1, 1,    1,    1,    1, None,    1,  1],
            [11,    1, 1,    1, 1,    1,    1,    1,    1, None,  1],
            [12,    1, 1,    1, 1,    1,    1,    1,    1, None,  1],
            [13,    1, 1,    1, 1,    1,    1,    1,    1,    1,  1],
        ])
        return pd.DataFrame(a)

    def test_get_not_null_vectors(self):
        df = self.get_big_df()
        total_rows = range(14)
        total_cols = range(11)

        attrs = [0, 4, 10]
        vectors = self.mss.get_not_null_vectors(attrs, df)
        self.assertEqual((14, 11), vectors.shape)
        self.assertEqual(total_rows, list(vectors[:, 0]))
        self.assertEqual(total_cols, list(vectors[0, :]))

        attrs = [0, 4, 10, 5]
        vectors = self.mss.get_not_null_vectors(attrs, df)
        self.assertEqual((12, 11), vectors.shape)
        self.assertEqual([0, 1, 2] + total_rows[5:], list(vectors[:, 0]))
        self.assertEqual(total_cols, list(vectors[0, :]))

        attrs = [9, 5]
        vectors = self.mss.get_not_null_vectors(attrs, df)
        self.assertEqual((10, 11), vectors.shape)
        self.assertEqual(total_rows[0:3] + total_rows[5:11] + total_rows[13:], list(vectors[:, 0]))
        self.assertEqual(total_cols, list(vectors[0, :]))

        attrs1 = [3, 5, 6, 7, 8]
        vectors1 = self.mss.get_not_null_vectors(attrs1, df)
        attrs2 = [3, 10, 8]
        vectors2 = self.mss.get_not_null_vectors(attrs2, df)
        total_rows = range(14)
        total_cols = range(11)
        self.assertTrue(np.array_equal(vectors1, vectors2))

    def test_get_null_vectors(self):
        df = self.get_big_df()
        total_rows = range(14)

        vectors = self.mss.get_null_vectors([1, 3], df)
        self.assertEqual(total_rows[1:11], list(vectors[:, 0]))

        vectors = self.mss.get_null_vectors([1], df)
        self.assertEqual(total_rows[3:11], list(vectors[:, 0]))

    def test_calc_item_matrix(self):
        df = self.get_big_df()
        attr_null = 1
        group_null = (9, 1)
        estimator = mock.Mock()
        estimator.fit.return_value = 1
        estimator.predict.return_value = -1

        self.mss.loader.hided_data = df
        result = self.mss.calc_item_matrix(attr_null, group_null, estimator)
        self.assertIsNone(result)

        group_null = (9,)
        result = self.mss.calc_item_matrix(attr_null, group_null, estimator)
        #
        # return_vector_not_null = ['vector_not_null']
        # get_not_null_mock = mock.Mock(return_value=return_vector_not_null)
        # self.mss.get_not_null_vectors = get_not_null_mock
        #
        # return_vector_null = ['vector_null']
        # get_null_mock = mock.Mock(return_value=return_vector_null)
        # self.mss.get_null_vectors = get_null_mock
        #
        # return_fill = ['fill']
        # get_fill_mock = mock.Mock(return_value=return_fill)
        # self.mss.fill_partial_data = get_fill_mock
        #
        # df_mock = mock.Mock()
        # original_df = pd.DataFrame
        # pd.DataFrame = mock.Mock(return_value=df_mock)
        #
        # return_score = mock.Mock()
        # calc_score_mock = mock.Mock(return_value=return_score)
        # self.mss.calc_score = calc_score_mock
        #
        # hided_data_mock = mock.Mock()
        # self.mss.loader.hided_data = hided_data_mock
        #
        # data_mock = mock.Mock()
        # self.mss.loader.data = data_mock
        # data_mock.shape = (2, 6)
        #
        # i = j = 1
        # self.assertIsNone(self.mss.calc_item_matrix(i, j, estimator))
        #
        # i = 0
        # result = self.mss.calc_item_matrix(i, j, estimator)
        # get_not_null_mock.assert_called_with([0, 2, 3, 4], self.mss.loader.hided_data)
        # get_null_mock.assert_called_with([i], self.mss.loader.hided_data)
        # get_fill_mock.assert_called_with(i, [2, 3, 4], return_vector_null, return_vector_not_null, self.mss.n)
        # calc_score_mock.assert_called_with(estimator, df_mock, range(5), bool(self.mss.seed))
        #
        # self.assertEqual(result, return_score.mean().abs())
        # pd.DataFrame = original_df

    def test_get_groups(self):
        df = self.get_big_df()
        expected_result_keys = [(), (1, 3, 5), (1, 6), (1, 6, 7), (1, 8), (3,), (9,)]
        result = self.mss.get_groups(df)
        self.assertEqual(sorted(result.keys()), expected_result_keys)

        idx = []
        for k, v in result.items():
            idx.extend(v)
        self.assertEqual(sorted(idx), range(df.shape[0]))
