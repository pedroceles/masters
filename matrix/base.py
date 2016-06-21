# -*- coding: utf-8 -*-
import numpy as np


class Matrix(object):
    def __init__(self, row_names, col_names):
        self.rows = np.array(row_names)
        self.cols = np.array(col_names)
        self.values = np.empty((len(self.rows), len(self.cols)))
        self.values[:] = np.NAN

    def __getitem__(self, *args):
        return self.values.__getitem__(*args)

    def __setitem__(self, *args):
        return self.values.__setitem__(*args)
