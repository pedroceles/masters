# -*- coding: utf-8 -*-
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt


def plot_abs_lines(data_file_path):
    df = pd.read_csv(data_file_path)
    df['diff'] = df.filled / df.full
    estimators = df.estimator.unique()
    estimators = sorted(estimators, key=lambda x: 0 if "Tree" in x or "Forest" in x else 1)
    fig, axe = plt.subplots(1, 1)
    for est in estimators:
        filtered = df[df.estimator == est]
        group_by = filtered.groupby('percent')
        mean = group_by.mean()
        std = group_by.std()
        line = mean.plot(y='filled', ax=axe, label=est, linewidth=3).lines[-1]
        color = line.get_c()
        upper_bound = mean.filled + std.filled
        upper_bound = upper_bound[upper_bound.notnull()]
        lower_bound = mean.filled - std.filled
        lower_bound = lower_bound[lower_bound.notnull()]
        axe.fill_between(mean.index, upper_bound, lower_bound, interpolate=True, alpha=0.2, color=color, linewidth=0)
    plt.show()


def plot_boxplot(data_file_path):
    df = pd.read_csv(data_file_path)
    df['times'] = df.filled / df.full
    estimators = df.estimator.unique()
    estimators = sorted(estimators, key=lambda x: 0 if "Tree" in x or "Forest" in x else 1)
    fig, axes = plt.subplots(2, 3)
    axes = list(chain(*axes))
    for est, ax in zip(estimators, axes):
        filtered = df[df.estimator == est]
        filtered = filtered[(filtered.percent * 100) % 10 == 0]
        filtered.boxplot('times', 'percent', ax=ax)
        ax.set_title(est)
    ylims = [ax.get_ylim() for ax in axes]
    miny = min(y[0] for y in ylims)
    maxy = max(y[1] for y in ylims)
    [ax.set_ylim((miny, maxy)) for ax in axes]
    fig.suptitle('')
    plt.show()
