# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

ERRORS = [u'naive', u'neigh', u'removed']


class Plotter(object):
    def __init__(self, df, classification=False):
        # self.df = df.loc[(slice(None), estimator), :].copy()
        self.df = df
        self.classification = classification
    #     self.add_columns()
    #
    # def add_columns(self):
    #     df = self.df
    #     top_groups = ['biased', 'random']
    #     compare_errors = [u'naive', u'neigh', u'removed']
    #
    #     for top_group in top_groups:
    #         complete = df.loc[:, (top_group, 'complete')]
    #         for error in compare_errors:
    #             errors_values = df.loc[:, (top_group, error)]
    #             df.loc[:, (top_group, 'd_{}'.format(error))] = 1 - (complete / errors_values)

    def pre_plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        return ax

    def post_plot(
        self, ax, show=True, title='', xlim=None, ylim=None,
        xlabel=None, ylabel=None
    ):
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if title:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if self.classification:
            ax.invert_yaxis()

        if show:
            plt.show()

    def plot_error_evolution(self, error, top_group, estimator, *args, **kwargs):
        from itertools import product
        ax = self.pre_plot(kwargs.pop('ax', None))
        dbs, estimators, percents = self.df.index.levels
        markers = ['*', '<', 'D', 'o', 'p']
        lss = ['-', '--', ':']
        line_props = product(markers, lss)
        for db in dbs:
            marker, ls = line_props.next()
            y = self.df.loc[(db, estimator), (top_group, error, 'mean')].values
            ax.plot(
                percents, y, label=db, color='k', ls=ls, marker=marker)
            ax.set_xticks(percents)
            ax.set_xticklabels(percents)
        plt.legend(loc='lower center', ncol=3)
        title = "Percent: {},\nMType: {},\nEst.: {}".format(
            error, top_group, estimator
        )
        self.post_plot(ax, title=title, *args, **kwargs)

    def plot_error_evolution_with_bar(
        self, error, top_group, estimator, dbs=None, *args, **kwargs
    ):
        from itertools import product
        ax = self.pre_plot(kwargs.pop('ax', None))
        dbs_, estimators_, percents = self.df.index.levels
        dbs = dbs or dbs_
        markers = ['*', '<', 'D', 'o', 'p']
        lss = ['-', '--', ':']
        line_props = product(markers, lss)
        for db in dbs:
            marker, ls = line_props.next()
            y = self.df.loc[(db, estimator), (top_group, error, 'mean')]
            y_err25 = y - self.df.loc[(db, estimator), (top_group, error, 'p25')]
            y_err75 = self.df.loc[(db, estimator), (top_group, error, 'p75')] - y
            ax.errorbar(
                percents, y, yerr=[y_err25, y_err75],
                label=db, ls=ls, marker=marker)
        plt.legend()
        self.post_plot(ax, *args, **kwargs)

    def plot_multiple_error_evolution(
        self, error=None, top_group=None, estimator=None, *args, **kwargs
    ):
        my_kwargs = {
            'error': error,
            'top_group': top_group,
            'estimator': estimator
        }
        if error is None:
            key = 'error'
            iterator = ['neigh', 'naive', 'removed']
        elif top_group is None:
            key = 'top_group'
            iterator = ['biased', 'random']
        elif estimator is None:
            key = 'estimator'
            dbs, estimators, percents = self.df.index.levels
            iterator = estimators
        kwargs.update(my_kwargs)
        fig, axes = plt.subplots(1, len(iterator))
        for ax, iteratee in zip(axes, iterator):
            kwargs[key] = iteratee
            self.plot_error_evolution(
                show=False, ax=ax, *args, **kwargs)
        plt.show()

    def plot_scatter_importance_error(
        self, percent, error, top_group, estimator, prop='important_gap',
        *args, **kwargs
    ):
        from imputation import all_runner
        runner_class = (
            all_runner.DBInfoClassificationImporter
            if self.classification else all_runner.DBInfoRegressionImporter
        )
        df_importance = runner_class().run()
        df_filtered = df_importance[
            (df_importance.loc[:, 'estimator'] == estimator)]
        importance_values = df_filtered[prop].values
        dbs, estimators, percents = self.df.index.levels
        # key = 'd_{}'.format(error)
        dbs, estimators, percents = self.df.index.levels
        ax = self.pre_plot(kwargs.pop('ax', None))
        y = self.df.loc[
            (slice(None), estimator, percent), (top_group, error, 'mean')].values
        ax.scatter(importance_values, y, c='k')
        ax.set_ylabel('Error Nonimprovement')
        ax.set_xlabel('Importance Range')
        if self.classification:
            ax.invert_xaxis()
        self.post_plot(ax, *args, **kwargs)

    def plot_errors_comparison(
        self, percent, top_group, estimator, *args, **kwargs
    ):

        # def _get_key(string):
        #     return 'd_{}'.format(string)

        dbs, estimators, percents = self.df.index.levels
        x = np.arange(len(dbs))

        # keys = map(_get_key, ERRORS)
        colors = ['#ffffff', '#bbbbbb', '#999999', '#555555', '#000000']
        width = 0.2
        ax = self.pre_plot(kwargs.pop('ax', None))
        for i, error in enumerate(ERRORS):
            x_ = x - (0.3 - i * width)
            values = self.df.loc[
                (slice(None), estimator, percent),
                (top_group, error, 'mean')
            ].values
            ax.bar(x_, values, label=error, color=colors[i], width=width)
        ax.legend()
        ax.set_xticklabels(dbs, rotation=45)
        ax.set_xticks(x)
        title = "Percent: {}, Missing Type: {}, Estimator: {}".format(
            percent, top_group, estimator
        )
        self.post_plot(ax, title=title, *args, **kwargs)

    def plot_missing_comparison(self, percent, estimator, *args, **kwargs):
        dbs, estimators, percents = self.df.index.levels
        x = np.arange(len(dbs))
        width = 0.8 / 6
        ax = self.pre_plot(kwargs.pop('ax', None))
        colors = ['#ffffff', '#bbbbbb', '#999999', '#555555', '#000000']
        bar_number = 0
        for j, error in enumerate(ERRORS):
            # key = 'd_{}'.format(error)
            color = colors[j]
            for i, missing_type in enumerate(['biased', 'random']):
                x_ = x - 0.4 + bar_number * width + (j / 30.0)
                values = self.df.loc[
                    (slice(None), estimator, percent),
                    (missing_type, error, 'mean')]
                label = u"{} {}".format(error, missing_type)
                ax.bar(
                    x_, values, label=label, color=color, width=width,
                    alpha=1 - i * 0.5
                )
                bar_number += 1
        plt.legend(loc='lower center')
        ax.set_xticklabels(dbs, rotation=45)
        ax.set_xticks(x)
        self.post_plot(ax, *args, **kwargs)

    def plot_estimator_comparison(
        self, percent, missing_type, error, *args, **kwargs
    ):
        dbs, estimators, percents = self.df.index.levels
        x = np.arange(len(dbs))
        width = 0.8 / 6
        ax = self.pre_plot(kwargs.pop('ax', None))
        colors = ['#ffffff', '#bbbbbb', '#999999', '#555555', '#000000']
        bar_number = 0
        # key = 'd_{}'.format(error)
        for i, estimator in enumerate(estimators):
            color = colors[i]
            x_ = x - 0.4 + bar_number * width
            values = self.df.loc[
                (slice(None), estimator, percent),
                (missing_type, error, 'mean')]
            label = estimator
            ax.bar(
                x_, values, label=label, color=color, width=width,
            )
            bar_number += 1
        plt.legend(loc='lower center')
        ax.set_xticklabels(dbs, rotation=45)
        ax.set_xticks(x)
        title = "Percent: {},\nMType: {},\niTreatment: {}".format(
            percent, missing_type, error
        )
        self.post_plot(ax, title=title, *args, **kwargs)

    def plot_multiple_estimator_comparison(
        self, error, percent, missing_type, *args, **kwargs
    ):
        my_kwargs = {
            'error': error,
            'missing_type': missing_type,
            'percent': percent
        }
        if error is None:
            key = 'error'
            iterator = ['neigh', 'naive', 'removed']
        elif missing_type is None:
            key = 'missing_type'
            iterator = ['biased', 'random']
        elif percent is None:
            key = 'percent'
            dbs, estimators, percents = self.df.index.levels
            iterator = percents
        kwargs.update(my_kwargs)
        fig, axes = plt.subplots(1, len(iterator))
        for ax, iteratee in zip(axes, iterator):
            kwargs[key] = iteratee
            self.plot_estimator_comparison(
                show=False, ax=ax, *args, **kwargs)
        plt.show()
