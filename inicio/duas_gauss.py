# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from testes_iniciais import gen_data, _get_nearest_indexes, get_mid_point

__all__ = ['GaussGen']


class GaussGen(object):
    """Classe para gerar dados"""
    def __init__(self, means=[[0, 0], [8, 8]], stds=[[1, 1], [1, 1]], covs=None, noise_ranges=[[-1, 1]], size=10000, n_neighbors=3):
        self.set_means(means)
        self.set_stds(stds)
        if not noise_ranges:
            noise_ranges = self.get_noise_range()
        self.set_covs(covs)
        self.noise_ranges = noise_ranges
        self.size = size
        self.gen_gauss_data()
        self.n_neighbors = n_neighbors

    def set_means(self, means):
        self.means = np.array(means)

    def set_stds(self, stds):
        self.stds = np.array(stds)

    def set_covs(self, covs):
        if covs:
            self.covs = np.array(covs)
        else:
            self.covs = None

    def get_noise_range(self):
        means = self.means
        stds = self.stds
        d_minus = means - stds
        d_plus = means + stds
        result = [[d_minus.min(), d_plus.max()]]
        return result

    def gen_gauss_data(self):
        means = self.means
        stds = self.stds
        covs = self.covs
        size = self.size
        noise_ranges = self.noise_ranges
        if covs is None:
            covs = np.array([np.diag(std) for std in stds])
            assert means.shape == covs.shape[:-1]
        size_per_gauss = size / len(means)
        data = np.empty((0, len(means[0]) + len(noise_ranges) + 1))  # +1 pra classe
        for classe, (m, c) in enumerate(zip(means, covs)):
            _data = gen_data(m, cov=c, zlims=noise_ranges, size=size_per_gauss)
            classes = np.zeros(_data.shape[0]).reshape(_data.shape[0], 1) + classe
            _data = np.append(_data, classes, axis=1)
            data = np.append(data, _data, axis=0)
        self.data = data

        return data

    def get_indexes(self, ordered_data, value, index=0):
        return _get_nearest_indexes(ordered_data[index], value, self.n_neighbors)

    def get_distances(self, get_same_class=True):
        if not hasattr(self, 'ordered_data'):
            indexes = np.array([range(0, len(self.data))]).T
            ordered_data = np.append(indexes, self.data, axis=1)
            self.ordered_data = np.array(sorted(ordered_data, key=lambda e: e[1]))
        ordered_data = self.ordered_data
        classes = [1]
        if get_same_class:
            classes = np.unique(ordered_data[:, -1])
        dists = []
        n = self.n_neighbors
        for c in classes:
            if get_same_class:
                ordered_data = self.ordered_data[self.ordered_data[:, -1] == c]
            for i, vals in enumerate(ordered_data):
                n_min = i - n if n <= i else 0
                n_max = i + n + 1
                ord_data = ordered_data
                split = ord_data[n_min:n_max, 0:3].copy()
                split[:, 2] = abs(split[:, 1] - vals[1])
                sort_split = np.array(sorted(split, key=lambda x: x[2]))
                indexes = sort_split[1: n + 1, 0].astype(int)
                point = get_mid_point(indexes, self.data[:, :-1])
                dists.append(abs(vals[1:-1] - point))
        dists = np.array(dists)
        self.dists = dists
        return dists

    def get_mult_distance(self, using_axis=[0], get_same_class=True):
        # using_axis são os backbones já selecionados que serão usados para o
        # cálculo dos n pontos mais próximos
        from scipy.spatial.distance import cdist
        # other_axis são os outros eixos
        other_axis = set(range(self.data.shape[1] - 1))
        other_axis = sorted(list(other_axis.difference(set(using_axis))))
        backbones = self.data[:, using_axis]
        # matriz 2 a 2 que calcula a distancia de cada vetor de backbones para o
        # outro
        dists_backbones_matrix = cdist(backbones, backbones)
        # guardando os índices de cada backbone, isso é necessário para quando
        # se usa classe
        indexes = np.arange(backbones.shape[0]).reshape(backbones.shape[0], 1)
        dists_backbones_matrix = np.append(indexes, dists_backbones_matrix, axis=1)  # Gravando os indices na coluna 0
        dists = []
        classes = [1]
        if get_same_class:
            classes = np.unique(self.data[:, -1])
        for c in classes:
            dists_backbones_matrix_to_use = dists_backbones_matrix
            if get_same_class:
                indexes_backbones = np.where(self.data[:, -1] == c)[0]  # pega os indices das linhas onde tal classe está
                dists_backbones_matrix_to_use = dists_backbones_matrix[indexes_backbones][:, np.insert(indexes_backbones + 1, 0, 0)]  # selecionando apenas os itens da matris
                # que tem os backbones com a devida classe
            else:
                indexes_backbones = indexes.reshape((indexes.shape[0], ))  # TODO Ver se isso atrapalha a performance. Tem que voltar
                # o indexes para a for de vetor sem ser vetor-coluna

            # nesse for, para cada backbone da classe irei pegar os n pontos
            # mais próximos calcular um ponto medio e calcular as distancias do
            # backbone e de cada eixo que se está avaliando (other axis)
            for i, backbone in enumerate(backbones[indexes_backbones]):
                # index dos pontos mais próximos na matriz reduzida, o que não
                # necessariamente equivale aos índices na matriz total (eles
                # foram gravados na coluna 0)
                indexes_sort = dists_backbones_matrix_to_use[i, 1:].argsort()[1:self.n_neighbors + 1]
                # indexes da matriz global, os que realmente serão usados para
                # calcular o mid-point
                real_indexes = dists_backbones_matrix_to_use[indexes_sort, 0].astype(int)
                # point é o ponto médio
                point = get_mid_point(real_indexes, self.data[:, :-1])
                # calculando a distancia euclideana entre os backbones e entre
                # cada eixo a ser avaliado. ATENÇÃO as duas distancias podem
                # ter sido medidos em Rns diferentes
                dist_backbones = [np.sqrt(np.sum((point[using_axis] - backbone) ** 2))]
                index_of_backbone = dists_backbones_matrix_to_use[i, 0]
                dist_other_points = np.sqrt((self.data[index_of_backbone, other_axis] - point[other_axis]) ** 2)
                # juntando as duas distâncias
                dist = dist_backbones + list(dist_other_points)
                dists.append(dist)
            print
            self.dists = np.array(dists)
        return self.dists

    def get_distances_means(self):
        return self.dists.mean(axis=0)

    def get_distances_stds(self):
        return self.dists.std(axis=0)

    def plot3views(self, axis=[0, 1, 2]):
        '''Plota gráficos xy, yz, xz dos dados'''

        def _print_matrix(matrix):
            s = '[{}]'
            pars = []
            for i in matrix:
                par = '({})'.format(','.join([str(j) for j in i]))
                pars.append(par)
            s = s.format(', '.join(pars))
            return s

        from itertools import combinations
        data = self.data
        fig, axes = plt.subplots(3, 1, sharex=True)
        ymin, ymax = np.Inf, -np.Inf
        for i, (x, y) in enumerate(combinations(axis, 2)):
            ax = axes[i]
            datax = data[:, x]
            datay = data[:, y]
            ymin = min(ymin, datay.min())
            ymax = max(ymax, datay.max())
            ax.scatter(datax, datay, c=data[:, -1], alpha=0.5, linewidth=0)
            ax.set_xlabel('x{}'.format(x))
            ax.set_ylabel('x{}'.format(y))
            # fig.suptitle(u'u: {}, stds: {} z:[{}, {}]'.format(
            #     _print_matrix(self.means),
            #     _print_matrix(self.stds),
            #     # self.noise_range[0],
            #     # self.noise_range[1],
            # ))
        for ax in axes:
            ax.set_ylim([ymin, ymax])
            # ax.set_aspect('equal')
        plt.show()

    @classmethod
    def run(cls, times=100, using_axis=[0], get_same_class=True, plot=True, title='', salvar_figura=None, **kwargs):
        from progressbar import ProgressBar
        pbar = ProgressBar(times).start()
        dist_means = []
        for i in range(times):
            pbar.update(i)
            g = cls(**kwargs)
            dists = g.get_mult_distance(using_axis, get_same_class=get_same_class)
            dist_means.append(dists.mean(axis=0))
        dist_means = np.array(dist_means)
        pbar.finish()
        if plot or salvar_figura:
            # positions = range(1, len(plot_axis) + 1)
            ax = plt.subplot()
            ax.boxplot(dist_means[:, 1:])  # , positions=positions)
            plt.title(title)
            if plot:
                plt.show()
            if salvar_figura:
                plt.savefig(salvar_figura)
            plt.clf()
        return dist_means
