# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from imputation import study
from datetime import datetime
import matplotlib.pyplot as plt


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


def run_errors_vs_percent_missing(
    runner_class, estimator=None, times=1, n_attrs=1, plot=False
):
    percents = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    if estimator:
        runner_class.estimator = estimator

    errors = []
    best_args = runner_class.get_important_from_file()[0:n_attrs]
    for percent in percents:
        print percent
        runner_class.percent_missing = percent
        runner = runner_class(best_args)
        me = runner.get_multiple_est_error_values(times)
        random = me.random
        errors.append(random.mean(axis=0))
    errors = np.array(errors)
    if plot:
        import matplotlib.pyplot as plt
        labels = ['neigh', 'naive', 'removed']
        for i, label in enumerate(labels):
            plt.plot(percents, errors[:, [i + 1]], label=label)
        plt.legend()
        plt.show()
    return errors


def same_estimator_comparison(runner_classes, estimator, times=1, n_attrs=1):
    errors = []
    for runner_class in runner_classes:
        print runner_class
        runner_class.estimator = estimator
        best_args = runner_class.get_important_from_file()[0:n_attrs]
        runner_class.percent_missing = 0.4
        runner = runner_class(best_args)
        me = runner.get_multiple_est_error_values(times)
        random = me.random
        errors.append(random.mean(axis=0))
    return errors


def plot_missing_conclusion(times=10, percent=0.2, num_instances=1000):
    from imputation import base_runner, fake_dbs
    from math import ceil

    final_table = []
    ns_attrs = [5, 10, 15, 20, 30]
    errors = []
    p25s = []
    p75s = []
    for i, n_attrs in enumerate(ns_attrs):
        db = fake_dbs.neigh_db(num_instances, n_attrs)
        runner = base_runner.MissingComparisonRegressionRunner([], values=db.values)
        attrs_to_use = int(ceil(n_attrs * percent))
        best_args = runner.get_important()[0: attrs_to_use]
        runner = base_runner.MissingComparisonRegressionRunner(best_args, values=db.values)
        runner.percent_missing = 0.4
        me = runner.mp_get_multiple_est_error_values(times)
        data = me.random
        mean = data.mean(0)
        p25 = np.percentile(data, 25, axis=0)
        p75 = np.percentile(data, 75, axis=0)
        errors.append(mean)
        p25s.append(p25)
        p75s.append(p75)
    errors = np.array(errors)
    ns = [int(ceil(n_attrs * percent)) for n_attrs in ns_attrs]
    methods = ["Neighbors", u"Naïve", "Remove"]

    def build_row(i):
        row = []
        for j in range(1, 4):
            error = errors[i][j] * 100
            p25 = p25s[i][j] * 100
            p75 = p75s[i][j] * 100
            row.append("{:.2f} ({:.2f} - {:.2f})".format(error, p25, p75))
            # row.append("{:.2f} - {:.2f}".format(p25, p75))
        return row

    def build_table():
        for i, error in enumerate(errors):
            final_table.append(build_row(i))
        columns = pd.MultiIndex.from_product([["MAPE (p25 - p75)"], methods])
        index = ["{}/{}".format(n, t) for n, t in zip(ns, ns_attrs)]
        df = pd.DataFrame(final_table, columns=columns, index=index)
        return df

    table = build_table()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    linestyles = ['-', '--', ':']
    for i in range(1, 4):
        errors[:, i] = errors[:, i] / errors[:, 0]
        ax.plot(
            ns, errors[:, i], color='k', linestyle=linestyles[i - 1],
            label=methods[i - 1]
        )
    ax.set_title('N. Instan.: {}. %: {}'.format(num_instances, percent * 100))
    x_ticklabels = ["{}/{}".format(n, t) for n, t in zip(ns, ns_attrs)]
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel("Attributes with missing data / Total attributes on database")
    ax.set_ylabel("Error Nonimprovement")
    plt.legend()
    plt.show()
    return table


def plot_ada_boost_conclusio(runner, classification=False):
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
    estimator = DecisionTreeRegressor() if not classification else DecisionTreeClassifier()
    meta_estimador = AdaBoostRegressor if not classification else AdaBoostClassifier
    runner.estimator = meta_estimador
    fig, axes = plt.subplots(1, 2, sharey=True)

    def plot(ax, errors, title):
        runner.plot_est_errors(ax, errors.mean(0))
        ax.set_title(title)
        ax.set_ylim((0, 1))

    # Simple tree
    me = runner.mp_get_multiple_est_error_values(1)
    plot(axes[0], me.random, "Simple Tree")

    runner.est_kwargs = {'base_estimator': estimator}
    me = runner.mp_get_multiple_est_error_values(1)
    plot(axes[1], me.random, "Normal Tree")
    plt.show()


def plot_categorical_imputation(times):
    import fake_dbs
    import base_runner

    fig, ax = plt.subplots(1, 1)
    db = fake_dbs.decision_tree_fake_point()
    r = base_runner.MissingComparisonRegressionRunner([0], [0, 1, 2, 3], values=db.values)
    me = r.mp_get_multiple_est_error_values(times)
    r.plot_est_errors(ax, me.random.mean(0), me.random.std(0))
    plt.show()
    return me


def plot_importance_correlation(rows=1000, n_attrs=4):
    import fake_dbs
    import base_runner

    fig, ax = plt.subplots(1, 1)
    factors = np.arange(1.25, 10, 0.25)
    for factor in factors:
        db = fake_dbs.importance_range_correlation(factor=factor)
        r = base_runner.MissingComparisonRegressionRunner([0], values=db.values)
        me = r.mp_get_multiple_est_error_values(1)
        scores = r.get_important(True)[1]
        importance_range = scores[0] / scores[-1]
        errors = me.random.mean(0)
        worse = errors[1] / errors[0]
        ax.scatter(importance_range, worse, color='k')
        ax.set_xlabel("Importance Range")
        ax.set_ylabel("Error Nonimprovement")
    plt.show()


def table_indifference_random_biased(db_type="regression", div=False):
    import df_transformer
    import glob
    fs = glob.glob(
        '/Users/pedroceles/dev/mestrado/dbs/{}/*/data/percent.pickle'.format(
            db_type
        )
    )

    df = df_transformer.DFAggregator(
        df_transformer.PercentDFTransformer, fs, div=False).get_df()
    table = []
    treatments = ['neigh', 'naive', 'removed']
    for treatment in treatments:
        errors = df.loc[:, (slice(None), treatment, 'mean')].values
        diff = abs(errors.T[1] - errors.T[0])
        if div:
            diff /= errors.T[1]
        table.append([
            diff.mean(),
            np.percentile(diff, 25),
            np.percentile(diff, 50),
            np.percentile(diff, 75),
        ])
    return pd.DataFrame(table, index=treatments, columns=[
        "Mean", "p25", "Median", "p75"])


def neigh_corr(times=10):
    import fake_dbs
    import base_runner
    import collections

    loosenesses = [0.01, 1, 5, 1000]
    # loosenesses = [0.01, 0.05, 0.10, 0.25, 0.5, 1, 5]
    correlations_means = []
    results = collections.defaultdict(list)
    for looseness in loosenesses:
        print "looseness", looseness
        corrs = []
        for time in range(times):
            df = fake_dbs.neighbors_difference(
                rows=2000, n_attrs=8, looseness=looseness)
            corrs.append(df.corr()[0][1:-1].mean())
            r = base_runner.MissingComparisonRegressionRunner(
                [0], values=df.values
            )
            errors = r.mp_get_multiple_est_error_values(times=1).random.mean(0)
            results[looseness].append(errors)
        correlations_means.append(np.array(corrs).mean())

    diffs = []
    for k, v in sorted(results.items()):
        arr = np.array(v)
        diff = (arr.T[1]) / arr.T[0]
        diffs.append(diff)

    plt.boxplot(diffs)
    ax = plt.axes()
    ax.set_xticklabels(["{:.2f}".format(x) for x in correlations_means])
    ax.set_xlabel(u"Correlation Mean")
    ax.set_ylabel(u"Error Nonimprovement")
    plt.show()
    return results


def get_winners_table():
    import pickle
    from imputation import df_transformer

    types = ["classification", "regression"]
    result = []
    profiles = {}
    for t in types:
        data = pickle.load(open(
            "/Users/pedroceles/dev/mestrado/study_data/percent_{}.pickle".format(t)
        ))
        df = df_transformer.DFAggregator(
            df_transformer.PercentDFTransformer, data=data).get_df()

        arg_fun = np.argmin if t == "regression" else np.argmax
        for kind_missing in ['random', 'biased']:
            filtered = df.loc[:, (kind_missing, slice(None), 'mean')]
            max_min = arg_fun(filtered.values, axis=1)
            result.append([
                t, kind_missing, len(max_min),
                sum(max_min == 0),
                sum(max_min == 1),
                sum(max_min == 2),
                sum(max_min == 3),
            ])

            treatments = ['neigh', u'naïve', 'no_rows', 'no_cols']
            for i in range(4):
                idx = max_min == i
                index = filtered[idx].index
                count = index.get_level_values(2).value_counts().sort_index()
                profiles[(t, kind_missing, treatments[i])] = count
    return result, profiles


def plot_winners_percent_profile(dict, db_type):

    def group():
        treatments = ['neigh', u'naïve', 'no_rows', 'no_cols']
        grouped = {}
        for t in treatments:
            kvs = [
                (k, v) for k, v in dict.items(
                ) if k[0] == db_type and k[2] == t
            ]
            for key, v in kvs:
                # group_key = 'removed' if key[-1] == 'removed' else 'inputed'
                group_key = key[-1]
                base_df = grouped.get(group_key)
                if base_df is None:
                    grouped[group_key] = v
                else:
                    grouped[group_key] += v
        return grouped

    group_dict = group()
    # colors = ['black', 'grey', '#cccccc']

    width = 0.02

    removed_values_rows = group_dict['no_rows']
    plt.bar(
        removed_values_rows.index + (-width / 2.0),
        removed_values_rows.values, label='No Rows', color='black', width=width)

    removed_values_cols = group_dict['no_cols']
    plt.bar(
        removed_values_cols.index + (-width / 2.0),
        removed_values_cols.values, label='No Cols', color='#666666', width=width,
        bottom=removed_values_rows.values
    )

    neigh_values = group_dict['neigh']
    plt.bar(
        neigh_values.index + (width / 2.0),
        neigh_values.values, label='Neighbors', color='#aaaaaa', width=width)

    naive_values = group_dict[u'naïve']
    plt.bar(
        naive_values.index + (width / 2.0),
        naive_values.values, label=u'Naïve', color='white', width=width,
        bottom=neigh_values.values
    )

    ax = plt.axes()
    ax.set_xticklabels(removed_values_rows.index)
    ax.set_xticks(removed_values_rows.index)
    ax.set_xlabel('%A')
    ax.set_ylabel('Amount')
    ax.set_title(db_type)

    plt.legend()
    plt.show()
