# -*- coding: utf-8 -*-
from __future__ import print_function
from matrix import study
import pickle
import os


def compare(hided_attrs, estimator, percent=0.3, pickled_file=None):
    if pickled_file:
        mss = pickle.load(pickled_file)
    else:
        mss = study.MatrixSingleStudy('~/Dropbox/Mestrado/Projeto/pesquisas/bike/data/treated.csv')
        mss.loader.hide_data_independent(*hided_attrs, percent=percent)
        backbones_fill = sorted(set(range(mss.n_attr)).difference(hided_attrs))
        mss.calc_matrix(estimator)
        mss.fill_data()
        mss.loader.fill_data(backbones_fill, hided_attrs, mss.n)
        mss.seed = 1
        pickle.dump(mss, open(os.path.join(os.path.dirname(__file__), 'matrix.pickle'), 'wb'))

    s1 = mss.calc_score(estimator, mss.filled_data, hided_attrs, True)
    s1 = abs(s1.mean())
    s2 = mss.calc_score(estimator, mss.loader.hided_data, hided_attrs, True)
    s2 = abs(s2.mean())

    return s1, s2


def full_comparison():
    import numpy as np
    ests = study.MatrixSingleStudy.get_estimators()
    mss = study.MatrixSingleStudy('~/Dropbox/Mestrado/Projeto/pesquisas/bike/data/treated.csv')
    attrs = range(mss.n_attr)
    f = open('~/Dropbox/Mestrado/Projeto/pesquisas/bike/data/analise_matrix.csv', 'w')
    for e in ests:
        print(e.__class__.__name__)
        for n in np.arange(0.5, 1, 0.2):
            print(n)
            from itertools import combinations
            for i in range(2, 7):
                combs = list(combinations(attrs, i))
                np.random.shuffle(combs)
                for hided_attrs in combs[:5]:
                    result = compare(hided_attrs, e, n)
                    print(hided_attrs, result, '### result')
                    print(e.__class__.__name__, end=';', file=f)
                    print(n, end=';', file=f)
                    print(hided_attrs, end=';', file=f)
                    print(result[0], end=';', file=f)
                    print(result[1], end='\n', file=f)
