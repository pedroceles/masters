# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

ERRORS = [u'naive', u'neigh', u'removed']


class Plotter(object):
    def __init__(self, df, estimator):
        self.df = df.loc[(slice(None), estimator), :].copy()
        self.estimator = estimator
        self.add_columns()

    def add_columns(self):
        df = self.df
        top_groups = ['biased', 'random']
        compare_errors = [u'naive', u'neigh', u'removed']

        for top_group in top_groups:
            complete = df.loc[:, (top_group, 'complete')]
            for error in compare_errors:
                errors_values = df.loc[:, (top_group, error)]
                df.loc[:, (top_group, 'd_{}'.format(error))] = 1 - (complete / errors_values)

    def plot_error_evolution(self, error, top_group):
        key = 'd_{}'.format(error)
        dbs, estimators, percents = self.df.index.levels
        fig, ax = plt.subplots(1, 1)
        for db in dbs:
            y = self.df.loc[(db, self.estimator), (top_group, key)].values
            ax.plot(percents, y, label=db)
            ax.set_xticks(percents)
            ax.set_xticklabels(percents)
        ax.legend()
        plt.show()

    def plot_errors_comparison(self, percent, top_group):

        def _get_key(string):
            return 'd_{}'.format(string)

        dbs, estimators, percents = self.df.index.levels
        x = np.arange(len(dbs))

        keys = map(_get_key, ERRORS)
        colors = ['blue', 'yellow', 'red', 'green']
        width = 0.2
        fig, ax = plt.subplots(1, 1)
        for i, key in enumerate(keys):
            x_ = x - (0.3 - i * width)
            values = self.df.loc[
                (slice(None), self.estimator, percent),
                (top_group, key)
            ].values
            ax.bar(x_, values, label=key, color=colors[i], width=width)
        ax.legend()
        ax.set_xticklabels(dbs)
        ax.set_xticks(x)
        plt.show()

    def plot_missing_comparison(self, percent):
        dbs, estimators, percents = self.df.index.levels
        x = np.arange(len(dbs))
        width = 0.8 / 6
        fig, ax = plt.subplots(1, 1)
        colors = ['blue', 'yellow', 'red', 'green']
        bar_number = 0
        for j, error in enumerate(ERRORS):
            key = 'd_{}'.format(error)
            color = colors[j]
            for i, missing_type in enumerate(['biased', 'random']):
                x_ = x - 0.4 + bar_number * width + (j / 30.0)
                values = self.df.loc[
                    (slice(None), self.estimator, percent),
                    (missing_type, key)]
                label = u"{} {}".format(error, missing_type)
                ax.bar(
                    x_, values, label=label, color=color, width=width,
                    alpha=1 - i * 0.5
                )
                bar_number += 1
        ax.legend()
        ax.set_xticklabels(dbs)
        ax.set_xticks(x)
        plt.show()
