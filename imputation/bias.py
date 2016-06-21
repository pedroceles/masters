# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class BiasedMissing(object):
    '''
    class to generate biased missing data
    '''
    def __init__(self, values, attrs, categorical_attrs, probs_function):
        self.values = values
        assert len(attrs) == len(probs_function)
        self.probs_function = probs_function
        self.attrs = attrs
        self.categorical_attrs = categorical_attrs
        self.non_categorical_attrs = list(set(range(self.values.shape[1])).difference(self.categorical_attrs))

    def hide_data(self, percent=None):
        self.hided_data = self.values.copy()
        if percent is not None:
            total_amount = int(percent * self.hided_data.shape[0])
        for attr, prob_function in zip(self.attrs, self.probs_function):
            x = self.hided_data[:, attr]
            probs = prob_function(x)
            assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be 0 <= p <= 1"
            randoms = np.random.random(x.shape[0])
            to_hide = randoms < probs
            if percent is not None:
                assert to_hide.sum() >= total_amount, "can't guarantee amount"
                indexes_true = np.where(to_hide)[0]
                to_hide = np.random.choice(indexes_true, total_amount, False)
            x[to_hide] = np.nan

    def get_complete_data(self):
        indexes = np.isnan(self.hided_data).any(1)
        return self.hided_data[~indexes]

    def get_missing_data(self, data):
        indexes = np.isnan(data).any(1)
        return data[indexes]

    def input_data_neighbors(self, n_neighbors, backbones):
        from sklearn.neighbors import NearestNeighbors
        from scipy.stats import mode

        def build_mask(attrs):
            zeros = np.zeros(self.values.shape[1], dtype=bool)
            zeros[attrs] = True
            return zeros
        mode_filter = build_mask(self.categorical_attrs)
        mean_filter = build_mask(self.non_categorical_attrs)

        filled_data = self.hided_data.copy()
        complete_data = self.get_complete_data()
        NN = NearestNeighbors(n_neighbors, metric='euclidean')
        NN.fit(complete_data[:, backbones])
        for i, vector in enumerate(filled_data):
            missing_data = np.isnan(vector)
            if not missing_data.any():
                continue
            distances, indexes = NN.kneighbors(vector[backbones])
            means = complete_data[indexes].mean(axis=1).reshape(complete_data.shape[1])
            modes = mode(complete_data[indexes], axis=1)[0].reshape(complete_data.shape[1])
            vector[missing_data & mean_filter] = means[missing_data & mean_filter]
            vector[missing_data & mode_filter] = modes[missing_data & mode_filter]
            filled_data[i] = vector
        return filled_data

    def plot_prob(self, index, ax=None):
        show = ax is None
        if show:
            fig, ax = plt.subplots(1, 1)
        attr = self.attrs[index]
        prob_function = self.probs_function[index]
        x = self.values[:, attr]
        x.sort()
        probs = prob_function(x)
        ax.plot(x, probs)
        ax.set_ylim(0, 1)
        if show:
            plt.show()


def p_factory_linear(min_value=0.0, max_value=10.0):
    delta = max_value - min_value

    def P(x):
        x_ = np.copy(x)
        x_[(x > max_value)] = 1
        x_[(x < min_value)] = 0
        between = [(x <= max_value) & (x >= min_value)]
        values = x_[between]
        values -= min_value
        values /= delta
        x_[between] = values
        return x_
    return P


def p_factory_gaussian(peaks, scales=None, max_prob=0.8):
    from scipy.stats import norm
    if scales is None:
        scales = [0.1] * len(peaks)

    def P(x):
        norms = [norm(peak, scale) for peak, scale in zip(peaks, scales)]
        y = sum(n.pdf(x) for n in norms)
        y = (y * max_prob) / y.max()
        return y

    return P
