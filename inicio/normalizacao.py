# -*- coding: utf-8 -*-
from duas_gauss import GaussGen
import numpy as np


def normalize(X, froms, tos):
    fmin, fmax = froms
    tmin, tmax = tos
    poly = np.polyfit(froms, tos, 1)
    return np.polyval(poly, X)


class GaussNorm(GaussGen):

    def __init__(self, *args, **kwargs):
        std_norm = kwargs.pop('std_norm', 4)
        super(GaussNorm, self).__init__(*args, **kwargs)
        cols = self.data.shape[1]
        for col in range(cols - 1):  # não pego a última coluna pq é a classe
            col_data = self.data[:, col]
            std = col_data.std()
            mean = col_data.mean()
            froms = [mean - std_norm * std, mean + std_norm * std]
            tos = [0, 1]
            norm_data = normalize(col_data, froms, tos)
            self.data[:, col] = norm_data
