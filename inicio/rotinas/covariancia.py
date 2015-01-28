# -*- coding: utf-8 -*-
import numpy as np
from inicio.normalizacao import GaussNorm


def gen_cov_matrix(size):
    rnd = np.random.randint(-10, 10, size=(size, size))
    diag = np.random.randint(1, 3, size=size)
    np.fill_diagonal(rnd, diag)  # diagonal tem que ser positiva
    rnd = (rnd + rnd.T) / 2  # isso é para deixar a matriz simétrica
    return rnd


def run(n_backbones=3, n_noise=1, n_gauss=1, times=100):
    cov_matrixes = [gen_cov_matrix(n_backbones) for x in range(n_gauss)]
    means = [np.random.randint(1, 10, n_backbones) for x in range(n_gauss)]
    noise_ranges = [[-1, 1] for x in range(n_noise)]
    for i in range(n_backbones - 2):
        print i
        nome_arquivo = '/home/pedroceles/Dropbox/Mestrado/Projeto/inicio/imagens/covariancia/cov_back_{}'.format(i + 1)
        titulo = "Backbones: {}".format(i + 1)
        GaussNorm.run(times, means=means, covs=cov_matrixes, using_axis=range(i + 1), noise_ranges=noise_ranges, n_neighbors=3, size=10000, title=titulo, salvar_figura=nome_arquivo, plot=False)
