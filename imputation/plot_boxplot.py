# -*- coding: utf-8 -*-
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt


def plot(data_file_path):
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
