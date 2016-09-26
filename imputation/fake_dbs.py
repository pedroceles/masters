import numpy as np
import pandas as pd


def decision_tree_fake_point(rows=1000, n_attrs=4):
    '''
    Creates a db with 4 attributes, the first one very important
    The last one just noise
    '''
    array = np.random.random((rows, n_attrs))
    noise = np.random.random((rows, n_attrs)) / 20

    result = 10 * array.T[0] + 1 * array.T[1] + 0.5 * array.T[2]

    attrs = array + noise
    for i in range(n_attrs):
        attrs[:, i] = (attrs[:, i] * 10).astype(int) / 10.0
    db_array = np.c_[attrs, result]
    return pd.DataFrame(db_array)


def neigh_db(rows=1000, attrs=4):
    array = np.random.random((rows, attrs))
    noise = np.random.random((rows, attrs)) / 20

    weights = np.random.random((attrs, 1)) * 10
    result = array.dot(weights)

    attrs = array + noise
    attrs[:, 0] = (attrs[:, 0] * 10).astype(int) / 10.0
    db_array = np.c_[attrs, result]
    return pd.DataFrame(db_array)


def importance_range_correlation(rows=1000, n_attrs=4, factor=10):
    array = np.random.random((rows, n_attrs))
    noise = np.random.random((rows, n_attrs)) / 20
    weights = np.array(
        [factor] + [1] * (n_attrs - 1)
    )

    attrs = array + noise
    result = array.dot(weights)
    db_array = np.c_[attrs, result]

    return pd.DataFrame(db_array)


def neighbors_difference(rows=1000, n_attrs=4, looseness=0.05):
    base = np.random.random((rows, 1))
    base = base.dot(np.ones((1, n_attrs)))
    others = np.random.random((rows, n_attrs))
    noise = np.random.random((rows, n_attrs)) / 20

    constants = np.random.random((1, n_attrs)) * looseness
    constants[0][0] = 0
    others_to_add = others * constants

    array = base + others_to_add
    weights = np.zeros(n_attrs)
    weights[0] = 1

    attrs = array + noise
    result = array.dot(weights)
    db_array = np.c_[attrs, result]

    return pd.DataFrame(db_array)
