# -*- coding: utf-8 -*-
import numpy as np
from imputation import study
from datetime import datetime


def run(n, source_file_name, results_file_name, **iss_init_kwargs):
    iss = study.ImputationStudy(source_file_name, **iss_init_kwargs)
    estimators = iss.get_estimators()
    attrs = np.arange(0, iss.n_attr - 1)
    f = open(results_file_name, 'w', 0)
    f.write('estimator,percent,full,filled\n')
    for i in range(n):
        iss.seed = np.random.randint(0, 2000)
        for percent in np.arange(0.1, 0.85, 0.1):
            iss.groups = {}
            print i, percent, datetime.now()
            iss.loader.hide_data_independent(*attrs, percent=percent)
            iss.fill_data()
            if iss.filled_data.isnull().any().any():
                import pudb; pudb.set_trace()  # XXX BREAKPOINT

            for estimator in estimators:
                full, filled = iss.compare(estimator, use_seed=True)
                f.write("{},{},{},{}\n".format(estimator.__class__.__name__, percent, full, filled))


def test_error_estimations(source_file_name, est, percent, step=0.02, **iss_init_kwargs):
    '''Test the regressions to see if it is possible to predict what
    the error would be if no data was missind'''
    from scipy import stats

    iss = study.ImputationStudy(source_file_name, **iss_init_kwargs)
    attrs = np.arange(iss.n_attr)
    iss.groups = {}
    iss.loader.hide_data_independent(*attrs, percent=percent)
    iss.fill_data()
    full, filled = iss.compare(est, use_seed=True)
    error_fill = [(percent, filled)]

    for new_percent in np.arange(percent + step, percent + step * 10, step):
        iss.loader.hide_extra(*attrs, percent=new_percent)
        iss.fill_data()
        full, filled = iss.compare(est, use_seed=True)
        error_fill.append((new_percent, filled))

    x, y = np.array(error_fill).T
    slope, intercept = stats.linregress(x, y)[:2]
    return full, intercept


def get_data_estimations(data_file_path, source_file_name, percent, step=0.02, **iss_init_kwargs):
    iss = study.ImputationStudy(source_file_name, **iss_init_kwargs)
    estimators = iss.get_estimators()

    result = []
    for i in range(10):
        for estimator in estimators:
            est_name = estimator.__class__.__name__
            print i, est_name
            full, intercept = test_error_estimations(source_file_name, estimator, percent, step, **iss_init_kwargs)
            result.append((est_name, full, intercept))

    import pickle
    pickle.dump(result, open(data_file_path, 'wb'))
    return result


def plot_estimations(pickle_file_path, estimators):
    import collections
    import pickle
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    colors = ['red', 'blue', 'green', 'black', 'gray', 'orange']
    ests_names = [e.__class__.__name__ for e in estimators]
    color_dict = dict(zip(ests_names, colors))
    data = np.array(pickle.load(open(pickle_file_path, 'rb'))).T
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim((0, 10 + len(ests_names) * 10))
    ax.set_xticks(range(5, 5 + 10 * len(ests_names), 10))
    ax.set_xticklabels(ests_names)
    total_points_counter = collections.Counter(data[0])
    points_counter = collections.Counter()

    for datum in data.T:
        est, full, intercept = datum
        full = float(full)
        intercept = float(intercept)
        points_counter[est] += 1
        count = points_counter[est]
        total = total_points_counter[est]
        base_x = ests_names.index(est)
        x_value = base_x * 10 + count / float(total) * 10
        ax.scatter(x_value, full, color=color_dict[est], marker='s')
        ax.scatter(x_value, intercept, color=color_dict[est], marker='^')
        ax.arrow(x_value, full, 0, intercept - full, color=color_dict[est], length_includes_head=True, head_length=0.001)

    max_y = max(data[1].astype(float).max(), data[2].astype(float).max())
    ax.set_ylim(0, max_y * 1.3)
    for index, est in enumerate(ests_names):
        filtered = np.array([x for x in data.T if x[0] == est])
        diff = abs(filtered[:, 1].astype(float) - filtered[:, 2].astype(float))
        ax.text(index * 10 + 1, max_y * 1.2, "std: {:>10.2f}\nmean: {:>6.2f}".format(diff.std(), diff.mean()), color=colors[index])
        # ax.text(index * 10 + 1, max_y * 1.1, "".format(diff.mean()), color=colors[index])

    legend_one = Line2D([], [], marker='s', linewidth=0, markeredgecolor='k', markerfacecolor='white', label=u"Erro")
    legend_two = Line2D([], [], marker='^', linewidth=0, markeredgecolor='k', markerfacecolor='white', label=u"Previsão")
    ax.legend(handles=[legend_one, legend_two], numpoints=1)
    return fig, ax
