# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa


def gen_data(means=[0, 0], std=[1, 1], cov=None, zlims=[[-1, 1]], size=10000):
    nz = len(zlims)
    nxs = len(means)
    if cov is None:
        cov = np.diag(std)
    xs = np.random.multivariate_normal(means, cov, size)
    z = np.random.random([size, nz])
    zlims = np.array(zlims).T
    z = z * (zlims[1, :] - zlims[0, :]) + zlims[0, :]
    shape = list(xs.shape)
    shape[1] += nz
    result = np.zeros(shape)
    result[:, :nxs] = xs
    result[:, nxs:] = z

    return result


def _get_nearest_indexes(x, value, n=1):
    from scipy.stats import rankdata
    delta = abs(x - value)
    # Enumerando o valor dos ranks que aí tenho o indíce de cada um,
    # Depois a lista de tuplas é ordenada pelo seus ranks, o resultado é então
    # transformado em uma np.array só de inteiros para poder ser usada como
    # index. Retorno os idices da posição 1 até n, poi a posição 0 será o
    # próprio valor de x pois a distancia dele para ele mesmo é zero.
    ranks = np.array(sorted(enumerate(rankdata(delta, 'ordinal')), key=lambda x: x[1]), dtype=int)
    return ranks[:, 0][1: n + 1]


def get_mid_point(indexes, data):
    return data[indexes].mean(axis=0)


def get_distances_neighbors(data, n=1):
    dists = []
    dt = data.T
    for i, vals in enumerate(data):
        _x = vals[0]
        indexes = _get_nearest_indexes(dt[0], _x, n)
        point = get_mid_point(indexes, data)
        dists.append(abs(vals - point))

    return np.array(dists)


def run_once(plot=False, ):
    data = gen_data()
    x, y, z = data.T
    if plot:
        size = len(x)
        fig = plt.figure()
        ax = fig.add_subplot('111', projection='3d')
        ax.scatter(x, y, z)

        index = np.random.randint(0, size)
        X, Y, Z = x[index], y[index], z[index]
        nearest_index = _get_nearest_indexes(x, X)
        nX, nY, nZ = x[nearest_index], y[nearest_index], z[nearest_index]
        ax.scatter((X, nX), (Y, nY), (Z, nZ), s=100, color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    x_dist, y_dist, z_dist = get_distances_neighbors(data).T
    y_mean, y_std = (y_dist.mean(), y_dist.std())
    z_mean, z_std = (z_dist.mean(), z_dist.std())
    return (y_mean, y_std), (z_mean, z_std)


def plot_result(runs=1000):
    from progressbar import ProgressBar
    pbar = ProgressBar(runs).start()
    y_means = []
    z_means = []
    y_stds = []
    z_stds = []

    for i in range(runs):
        pbar.update(i)
        (y_mean, y_std), (z_mean, z_std) = run_once()
        y_means.append(y_mean)
        z_means.append(z_mean)
        y_stds.append(y_std)
        z_stds.append(z_std)
    pbar.finish()
    ax = plt.subplot()

    ax.boxplot((y_means, z_means), positions=[1, 2])
    ax.set_xticklabels(['y', 'z'])
    # plt.boxplot(y_means, label="y_means")
    # plt.boxplot(z_means, label="z_means")
    # plt.plot(y_stds, label="y_stds")
    # plt.plot(z_stds, label="z_stds")

    plt.legend()
    plt.show()
