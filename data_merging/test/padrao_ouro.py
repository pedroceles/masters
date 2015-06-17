import unittest


class TestPadraoOuro(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_all_combinations(self):
        from data_merging.padrao_ouro import all_combinations
        result = all_combinations([1, 2, 3])
        self.assertEqual(len(result), 7)
        expected_result = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        self.assertEqual(sorted(expected_result), sorted(result))

    def test_make_padrao(self):
        import os
        import tempfile
        import shutil
        import mock
        import data_merging.padrao_ouro as po
        from data_merging.padrao_ouro import make_padrao
        from data_merging.study import DataSingleStudy

        m = mock.Mock(return_value=[[1], [2], [3], [1, 2]])
        po.all_combinations = m

        tempdir = tempfile.mkdtemp()
        fname = '/home/pedroceles/Dropbox/Mestrado/Projeto/pesquisas/wine/data/winequality/winequality-red.csv'
        dss = DataSingleStudy(fname, sep=';')
        make_padrao(dss, tempdir)
        m.assert_called_with(range(11))
        for est in dss.get_estimators():
            fname = os.path.join(tempdir, 'ouro_' + est.__class__.__name__.lower() + '.txt')
            self.assertTrue(os.path.isfile(fname))
            f = open(fname, 'r')
            lines = f.readlines()
            self.assertEqual(len(lines), 4)
            line = lines[0]
            self.assertIsInstance(eval(line.split(';')[0]), list)
            self.assertIsInstance(eval(line.split(';')[1]), float)
        shutil.rmtree(tempdir)
