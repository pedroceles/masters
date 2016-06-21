# -*- coding: utf-8 -*-
import unittest
import mock
from data_merging.study import DataMultiStudy as DMS


class TestDataMultiStudy(unittest.TestCase):

    """"""

    def setUp(self):
        import tempfile
        self.f = tempfile.TemporaryFile('wr')
        f = self.f
        LINE_TEMPLATE = '{},{},{},{}\n'
        f.write(LINE_TEMPLATE.format('col1', 'col2', 'col3', 'result'))
        f.write(LINE_TEMPLATE.format(1, 1, 1, 1))
        f.write(LINE_TEMPLATE.format(1, 1, 0, 1))
        f.write(LINE_TEMPLATE.format(0, 1, 0, 0))
        f.write(LINE_TEMPLATE.format(0, 0, 0, 0))
        f.write(LINE_TEMPLATE.format(1, 0, 0, 0))
        f.seek(0)

        self.dms = DMS(f)

    def tearDown(self):
        pass

    def test_assert_basic(self):
        ds = self.dms.get_ds()
        self.assertEqual(ds.loader.data.shape, (5, 4))

    def test_select_most_important(self):
        from itertools import combinations
        mock_return = {
            (0, 2): 0.3,
            (1, 2): 0.3,
        }
        comparison = mock.Mock(return_value=mock_return)
        self.dms.make_multi_backbones_comparison = comparison
        pre_selected = [2]
        result = self.dms.select_most_important(pre_selected=pre_selected)
        self.assertEqual(sorted(comparison.call_args[0][0]), [pre_selected + [0], pre_selected + [1]])
        expected_keys = [0, 1]
        for k, v in result:
            self.assertIn(k, expected_keys)
            self.assertAlmostEqual(0.3, v)
        combs = sorted(combinations([0, 1, 2], 2))
        comparison.return_value = dict([(x, 0.3) for x in combs])

        result = self.dms.select_most_important()
        self.assertEqual(map(lambda x: list(x), combs), sorted(comparison.call_args[0][0]))
        expected_keys = [0, 1, 2]
        for k, v in result:
            self.assertIn(k, expected_keys)
            self.assertAlmostEqual(0.3, v)

        combs = sorted(combinations([0, 2], 1))
        comparison.return_value = dict([(x, 0.3) for x in combs])
        result = self.dms.select_most_important(eliminate=[1])
        self.assertEqual(map(lambda x: list(x), combs), sorted(comparison.call_args[0][0]))
        expected_keys = [0, 2]
        for k, v in result:
            self.assertIn(k, expected_keys)
            self.assertAlmostEqual(0.3, v)

    def test_select_features(self):
        import numpy as np
        estimators = self.dms.get_ds().get_estimators()

        self.call_counts = 0

        def calc_error_return(*args, **kwargs):
            self.call_counts += 1
            if self.call_counts <= 2:
                return - np.array([100 - self.call_counts])
            return np.array([-100])

        ds = mock.Mock()
        ds.calc_score = mock.Mock(side_effect=calc_error_return)
        get_ds = mock.Mock(return_value=ds)
        self.dms.get_ds = get_ds
        result = self.dms.select_features(estimator=estimators[0])
        self.assertIn(estimators[0], ds.calc_score.call_args[0])
        self.assertEqual(3, ds.calc_score.call_count)
        self.assertIsInstance(result, list)

if __name__ == '__main__':
    unittest.main()
