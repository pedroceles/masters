# -*- coding: utf-8 -*-
from collections import namedtuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn import tree
from sklearn import naive_bayes

CLASSIFIER_ESTIMATORS = [ensemble.RandomForestClassifier, ensemble.ExtraTreesClassifier, tree.DecisionTreeClassifier, tree.ExtraTreeClassifier, naive_bayes.GaussianNB]
REGRESSION_ESTIMATORS = [ensemble.RandomForestRegressor, ensemble.ExtraTreesRegressor, tree.DecisionTreeRegressor, tree.ExtraTreeRegressor]

GeneralValues = namedtuple('GeneralValues', ['X', 'y', 'X_train', 'X_test', 'y_train', 'y_test', 'backbones'])
MissingValues = namedtuple('MissingValues', ['biased', 'random'])
FilledValues = namedtuple('FilledValues', ['neigh', 'naive'])
ErrorValues = namedtuple('ErrorValues', ['complete', 'neigh', 'naive', 'no_missing', 'no_attrs'])
Values = namedtuple('Values', ['filled_values', 'error_values'])
AllValues = namedtuple('AllValues', ['gen_values', 'biased', 'random'])
MultipleErrorValues = namedtuple('MultipleErrorValues', ['biased', 'random'])


def save_data(data, f_path):
    import pickle
    pickle.dump(data, open(f_path, 'wb'))


def mape(y_true, y_pred):
    diff = abs(np.array(y_true) - np.array(y_pred))
    div = diff / y_true.astype(float)
    return div.mean()


class BaseRunner(object):
    def __init__(self, source_data_file=None, values=None):
        self._values = values if values is not None else self.load_values(source_data_file)

    def load_values(self, source_data_file):
        import pandas as pd
        return pd.read_csv(source_data_file, header=None).values

    @property
    def values(self):
        return self._values.astype(float)

    def run(self):
        raise NotImplementedError()

    @staticmethod
    def calc_score(self, X, y):
        raise NotImplementedError()


from sklearn.feature_selection import chi2


class MissingComparisonRegressionRunner(BaseRunner):
    test_split = 0.4
    estimator = tree.DecisionTreeRegressor
    classification = False
    percent_missing = None
    seed = None

    def __init__(self, attrs_missing, categorical_attrs=None, *args, **kwargs):
        super(MissingComparisonRegressionRunner, self).__init__(*args, **kwargs)
        self.categorical_attrs = categorical_attrs or []
        self.attrs_missing = attrs_missing
        self.metric_name = "Accu." if self.classification else "MAPE"
        self.train_test_data = None

    def update_seed(self):
        self.seed = int(np.random.random() * 1000000)

    def init_bm(self, values):
        from bias import BiasedMissing
        ps = self.get_p_functions()
        self.bm = BiasedMissing(values.copy(), self.attrs_missing, self.categorical_attrs, ps)

    def get_p_functions(self, peaks_list=None, scales_list=None, max_prob=0.95):
        from bias import p_factory_gaussian
        if peaks_list is None:
            peaks_list = [[x.mean()] for x in self.values[:, self.attrs_missing].T]
        if scales_list is None:
            scales_list = [[x.std()] for x in self.values[:, self.attrs_missing].T]

        assert len(peaks_list) == len(scales_list) == len(self.attrs_missing)
        ps = [p_factory_gaussian(peaks, scales, max_prob) for peaks, scales in zip(peaks_list, scales_list)]
        return ps

    def init_estimator(self, **kwargs):
        _kwargs = kwargs or getattr(self, 'est_kwargs', {}) or {}
        return self.estimator(**_kwargs)

    def miss_data(self):
        self.bm.hide_data(self.percent_missing)
        amount_nan = np.isnan(self.bm.hided_data).sum(axis=0)
        random_missing = self.bm.values.copy()
        indexes = np.arange(random_missing.shape[0])
        for attr, amount in enumerate(amount_nan):
            if amount:
                indexes_to_none = np.random.choice(indexes, amount, False)
                random_missing[indexes_to_none, [attr]] = None
        assert (amount_nan == np.isnan(random_missing).sum(axis=0)).all()
        return self.bm.hided_data, random_missing

    def get_compare_error_hist(self, attr, bins=10):
        gen_values = self.get_gen_values()
        X = gen_values.X
        bins = np.histogram(X[:, attr], bins)[1]
        range_ = abs(bins[0] - bins[-1])
        bins[-1] += range_ * .01
        bins[0] -= range_ * .01
        missing_values = self.get_missing_values(gen_values.X_train)
        filled_values_biased = self.get_filled_values(
            missing_values.biased, gen_values.backbones)
        filled_values_random = self.get_filled_values(
            missing_values.random, gen_values.backbones)

        df_trained = pd.DataFrame()
        df_trained['x'] = gen_values.X_train[:, attr]
        df_trained['x_biased'] = filled_values_biased.neigh[:, attr]
        df_trained['x_random'] = filled_values_random.neigh[:, attr]
        df_trained['x_missing_biased'] = np.isnan(
            missing_values.biased[:, attr])
        df_trained['x_missing_random'] = np.isnan(
            missing_values.random[:, attr])
        df_trained['x_group'] = pd.cut(df_trained['x'], bins)

        gb_count_missing = df_trained.groupby('x_group')[
            'x_missing_biased', 'x_missing_random'].aggregate([np.sum, len])

        assert gb_count_missing['x_missing_biased', 'sum'].sum() == \
            gb_count_missing['x_missing_random', 'sum'].sum()

        est = self.init_estimator()
        df = pd.DataFrame()
        df['x'] = gen_values.X_train[:, attr]
        df['x_group'] = pd.cut(df['x'], bins)
        df['y'] = gen_values.y_train

        self.update_seed()
        est = self.init_estimator()
        np.random.seed(self.seed)
        est.fit(filled_values_biased.neigh, gen_values.y_train)
        df['y_pred_biased'] = est.predict(gen_values.X_train)

        np.random.seed(self.seed)
        est.fit(filled_values_random.neigh, gen_values.y_train)
        df['y_pred_random'] = est.predict(gen_values.X_train)

        df['ape_biased'] = abs(
            df['y_pred_biased'] - gen_values.y_train) / gen_values.y_train
        df['ape_random'] = abs(
            df['y_pred_random'] - gen_values.y_train) / gen_values.y_train

        gb = df.groupby('x_group')['ape_biased', 'ape_random'].mean()
        result_gb = pd.concat([gb, gb_count_missing], axis=1)
        return result_gb, df[['ape_biased', 'ape_random']].mean()

    def treat_table_error_hist(self, gb):
        new_df = pd.DataFrame()
        new_df['percent_biased'] = gb['x_missing_biased', 'sum'] / \
            gb['x_missing_biased', 'len']
        new_df['percent_random'] = gb['x_missing_random', 'sum'] / \
            gb['x_missing_random', 'len']

        new_df['biased_sum'] = gb['x_missing_biased', 'sum']
        new_df['random_sum'] = gb['x_missing_random', 'sum']

        new_df['ape_biased'] = gb['ape_biased']
        new_df['ape_random'] = gb['ape_random']
        return new_df

    def run_treat(self, attr, times, bins=10):
        dfs = []
        dfs_error = []
        for i in range(times):
            df, df_error = self.get_compare_error_hist(attr, bins=bins)
            dfs.append(self.treat_table_error_hist(df))
            dfs_error.append(df_error)
        df = pd.concat(dfs)
        df = df.groupby(df.index).mean()
        return df, df_error

    def set_train_test_data(self, data):
        self.train_test_data = data

    def get_train_test_data(self):
        from sklearn.cross_validation import train_test_split

        if self.train_test_data:
            return self.train_test_data

        values = np.copy(self.values)
        X, y = values[:, :-1], values[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        return (X, y, X_train, X_test, y_train, y_test)

    def get_gen_values(self):
        X, y, X_train, X_test, y_train, y_test = self.get_train_test_data()

        backbones = set(range(X_train.shape[1]))
        backbones = list(backbones.difference(self.attrs_missing))
        return GeneralValues(X, y, X_train, X_test, y_train, y_test, backbones)

    def get_missing_values(self, X_train):
        train_size = X_train.shape[0]
        indexes_to_miss = np.random.choice(np.arange(train_size),
                                           int(train_size * 0.9),  # Tenho que deixar 10% livre de missing, para que o vizinho mais próximo ache dados completos
                                           False
                                           )
        X_train_to_miss = X_train[indexes_to_miss, :]

        self.init_bm(X_train_to_miss)
        pre_biased_missing, pre_random_missing = self.miss_data()

        biased_missing = X_train.copy()
        biased_missing[indexes_to_miss, :] = pre_biased_missing

        random_missing = X_train.copy()
        random_missing[indexes_to_miss, :] = pre_random_missing

        return MissingValues(biased_missing, random_missing)

    def get_filled_values(self, missing_values, backbones):
        neigh, naive = self.fill_data(missing_values, backbones)
        return FilledValues(neigh, naive)

    def get_est_error_values(self, est, gen_values, filled_values, missing_values):
        X_train, y_train, X_test, y_test = gen_values.X_train, gen_values.y_train, gen_values.X_test, gen_values.y_test
        indexes_no_missing = ~np.isnan(missing_values).any(1)
        error_complete = self.run_estimator(est, X_train, y_train, X_test, y_test)
        error_neigh = self.run_estimator(est, filled_values.neigh, y_train, X_test, y_test)
        error_naive = self.run_estimator(est, filled_values.naive, y_train, X_test, y_test)

        X_no_missing = X_train[indexes_no_missing]
        y_no_missing = y_train[indexes_no_missing]
        error_no_missing = self.run_estimator(est, X_no_missing, y_no_missing, X_test, y_test)

        X_train_no_attrs = np.delete(X_train, self.attrs_missing, 1)
        X_test_no_attrs = np.delete(X_test, self.attrs_missing, 1)
        error_no_attrs = self.run_estimator(est, X_train_no_attrs, y_train, X_test_no_attrs, y_test)
        return ErrorValues(error_complete, error_neigh, error_naive, error_no_missing, error_no_attrs)

    def get_values(self, est, gen_values, missing_values):
        filled_values = self.get_filled_values(missing_values, gen_values.backbones)
        error_values = self.get_est_error_values(est, gen_values, filled_values, missing_values)
        return Values(filled_values, error_values)

    def get_all_values(self, gen_values):
        np.random.seed()
        seed_value = self.seed
        if not self.seed:
            self.update_seed()
        print self.seed, '######'
        missing_values = self.get_missing_values(gen_values.X_train)

        self.est = self.init_estimator()
        biased = self.get_values(self.est, gen_values, missing_values.biased)
        random = self.get_values(self.est, gen_values, missing_values.random)

        self.seed = seed_value
        return AllValues(gen_values, biased, random)

    def get_multiple_est_error_values(self, times=100):
        biased_errors = []
        random_errors = []
        gen_values = self.get_gen_values()
        for i in xrange(times):
            print i, self.seed
            all_vals = self.get_all_values(gen_values)
            biased_errors.append(all_vals.biased.error_values)
            random_errors.append(all_vals.random.error_values)
        return MultipleErrorValues(np.array(biased_errors), np.array(random_errors))

    def mp_get_multiple_est_error_values(self, times=100):
        import multiprocessing as mp
        from mp import single_runner
        m = mp.Manager()
        result_q = m.Queue()
        pool = mp.Pool()
        method_name = 'get_all_values'
        gen_values = self.get_gen_values()
        pool.map_async(
            single_runner, zip(
                [self] * times, [method_name] * times, [result_q] * (times),
                [gen_values] * times
            )
        )
        pool.close()
        results = []
        for i in range(times):
            print 'Getting', i
            results.append(result_q.get())
            print 'Got', i
        biased_errors = [r.biased.error_values for r in results]
        random_errors = [r.random.error_values for r in results]
        return MultipleErrorValues(np.array(biased_errors), np.array(random_errors))

    def plot_multiple_est_error_values(self, multiple_errors, axes=None):
        show = axes is None
        if show:
            fig, axes = plt.subplots(1, 2, sharey=True)
        for data, ax, title in zip(
            multiple_errors, axes, [u"Bias", u"Random"]
        ):
            mean, std = data.mean(0), data.std(0)
            self.plot_est_errors(ax, mean, std, title=title)
        if show:
            fig.suptitle(self.estimator.__name__)
            plt.show()

    def plot(self):
        gen_values = self.get_gen_values()
        all_vals = self.get_all_values(gen_values)

        fig, axes_rows = plt.subplots(2, len(self.attrs_missing) + 1)
        values_group = [all_vals.biased, all_vals.random]
        X_train = all_vals.gen_values.X_train
        for values, axes, title in zip(
            values_group, axes_rows, [u"Viés", u"Randômico"]
        ):
            filled_values = values.filled_values
            axes_hist = axes[:-1]
            for attr_missing, ax_hist in zip(self.attrs_missing, axes_hist):
                self.plot_histogram(ax_hist, attr_missing, X_train, filled_values)
            ax_bar = axes[-1]
            error_values = values.error_values
            self.plot_est_errors(ax_bar, error_values, title=title)
        ax_bars = np.array(axes_rows)[:, -1]
        max_lim = max(ax.get_ylim()[1] for ax in ax_bars)
        [ax.set_ylim((0, max_lim)) for ax in ax_bars]
        fig.suptitle(self.estimator.__name__)
        plt.show()

    def plot_missing_hist_example(self, attr):
        assert attr in self.attrs_missing
        from matplotlib.ticker import FormatStrFormatter

        fig, axes = plt.subplots(1, 2)
        ax_hist, ax_prob = axes
        gen_values = self.get_gen_values()
        missing_values = self.get_missing_values(gen_values.X_train)
        range_ = gen_values.X_train[:, attr].min(), gen_values.X_train[:, attr].max()
        hist_complete, edges = np.histogram(gen_values.X_train[:, attr], range=range_)
        hist_biased, edges = np.histogram(missing_values.biased[:, attr], range=range_)
        hist_random, edges = np.histogram(missing_values.random[:, attr], range=range_)

        min_value, max_value = edges.min(), edges.max()

        edges = edges[:-1]
        width = abs(edges[0] - edges[1]) / 2.0
        ax_hist.bar(edges - width, hist_complete, width * 2, label="Original", color='#222222')
        ax_hist.bar(edges - width, hist_biased, width, label=u"Biased", color='grey')
        ax_hist.bar(edges, hist_random, width, label=u"Random", color='white')
        ax_hist.legend()
        ax_hist.set_xlim(edges[0] - width, edges[-1] + width)
        ax_hist.set_xticks(edges)
        ax_hist.set_xlabel('Attribute value')
        ax_hist.set_ylabel('Number of occurrences')
        ax_hist.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_hist.set_title('Attribute Value Distribution')

        attr_index = self.attrs_missing.index(attr)
        prob_f = self.get_p_functions()[attr_index]
        x_values = np.linspace(min_value, max_value, 1000)
        ax_prob.plot(x_values, prob_f(x_values), color='black')
        ax_prob.set_xticks(edges)
        ax_prob.set_xlabel('Attribute value')
        ax_prob.set_ylabel('Probability to be missing')
        ax_prob.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_prob.set_title('Attribute Value Missing Probability')

        plt.show()

    def plot_important(self):
        gen_values = self.get_gen_values()
        attrs = range(0, self.values.shape[1] - 1)
        self.update_seed()
        errors = []
        for exclude_attr in attrs:
            use_attrs = attrs[:exclude_attr] + attrs[exclude_attr + 1:]
            getter = [slice(None), use_attrs]
            est = self.init_estimator()
            np.random.seed(self.seed)
            error = self.run_estimator(
                est, gen_values.X_train[getter], gen_values.y_train,
                gen_values.X_test[getter], gen_values.y_test
            )
            errors.append(error)

        fig, ax = plt.subplots(1, 1)
        ax.bar(np.array(attrs) - 0.4, errors)
        ax.set_xticklabels(attrs)
        ax.set_xticks(attrs)
        ax.set_xlabel('Atributo Faltante')
        plt.show()

    def plot_important_from_file(self):
        i, scores = self.get_important_from_file(True)
        fig, ax = plt.subplots(1, 1)
        x = np.arange(len(i))
        width = 0.8
        ax.bar(x - 0.4, scores, color='gray', width=width)
        ax.set_xticks(x)
        ax.set_xticklabels(i)
        ax.set_xlabel("Attribute")
        ax.set_ylabel("MAPE" if not self.classification else "Ac")
        plt.show()

    def fill_data(self, missing_values, backbones):
        from sklearn.preprocessing import Imputer
        imp_mean = Imputer(strategy='mean')
        imp_mode = Imputer(strategy='most_frequent')

        self.bm.hided_data = missing_values
        filled_values = self.bm.input_data_neighbors(20, backbones)

        imputed_values = missing_values.copy()
        for attr_missing in self.attrs_missing:
            if attr_missing not in self.categorical_attrs:
                imputed_values[:, [attr_missing]] = imp_mean.fit_transform(imputed_values[:, [attr_missing]])
            else:
                imputed_values[:, [attr_missing]] = imp_mode.fit_transform(imputed_values[:, [attr_missing]])
        return filled_values, imputed_values

    def plot_histogram(self, ax_hist, attr_missing, original_values, filled_values):
        original_color = 'blue'
        neigh_color = 'green'
        naive_color = 'red'
        original_data = original_values[:, attr_missing]
        range_ = (original_data.min(), original_data.max())
        ax_hist.hist(original_data, range=range_, label="Original", color=original_color)
        ax_hist.hist(filled_values.neigh[:, attr_missing], range=range_, alpha=0.5, label="Imputado Vizinhos", color=neigh_color)
        ax_hist.hist(filled_values.naive[:, attr_missing], range=range_, alpha=0.5, label=u"Imputado Naïve", color=naive_color)
        ax_hist.set_yticklabels([])
        ax_hist.set_title(attr_missing)

    def plot_est_errors(self, ax_bar, error_values, yerr_values=None, title=""):
        import textwrap
        original_color = 'Black'
        neigh_color = '#444444'
        naive_color = '#888888'
        no_missing_color = '#bbbbbb'
        no_attrs_color = '#ffffff'
        colors = [original_color, neigh_color, naive_color, no_missing_color, no_attrs_color]

        range_error = np.arange(5) + 1
        width = 0.75
        ax_bar.bar(
            range_error - width / 2., error_values, width, color=colors,
            yerr=yerr_values, ecolor='black'
        )
        ax_bar.set_xticks([1, 2, 3, 4, 5])
        xticklabels_text = [
            "Original", "Neighbors", u"Naïve", "No Rows", "No Cols"
        ]
        xticklabels = [textwrap.fill(text, 10) for text in xticklabels_text]
        ylim_max = 0
        for i, error in enumerate(error_values):
            y = error * 1.1
            if yerr_values is not None:
                y += yerr_values[i]
            ylim_max = max(ylim_max, y)
            ax_bar.text(i + 1, y, "{:.2f}".format(error))

        ax_bar.set_ylim(top=ylim_max * 1.1)
        ax_bar.set_xticklabels(xticklabels)
        ax_bar.set_ylabel(self.metric_name)
        ax_bar.set_title(title)

    def plot_corr(self, attr, ax=None):
        show = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        df = pd.DataFrame(self.values[:, :-1])
        df.corr()[attr].plot(kind='bar', ax=ax)
        if show:
            plt.show()

    def plot_best_args(self, f=chi2):
        data = np.array(self.get_best_args(f, True))
        fig, ax = plt.subplots(1, 1)
        ax.bar(data[0] - 0.4, data[1])
        ax.set_xlabel('Atributos')
        ax.set_ylabel('Score')
        ax.set_xlim(left=-1)
        ax.set_xticks(data[0])
        ax.set_xticklabels(data[0].astype(int))
        plt.show()

    @staticmethod
    def train_estimator(est, X_train, y_train):
        est.fit(X_train, y_train)

    def calc_score(self, est, X_test, y_test):
        from sklearn.metrics import accuracy_score
        metric = accuracy_score if self.classification else mape
        y_pred = est.predict(X_test)
        return metric(y_test, y_pred)

    def run_estimator(self, est, X_train, y_train, X_test, y_test):
        np.random.seed(self.seed)
        self.train_estimator(est, X_train, y_train)
        return self.calc_score(est, X_test, y_test)

    def get_best_args(self, metric=chi2, return_scores=False):
        from sklearn.feature_selection import SelectKBest
        sel = SelectKBest(metric, 'all')
        sel.fit(self.values[:, :-1], self.values[:, -1])
        if not return_scores:
            return sel.scores_.argsort()[-1::-1]
        else:
            return sel.scores_.argsort()[-1::-1], sorted(sel.scores_)[-1::-1]

    def get_important(self, return_scores=False):
        gen_values = self.get_gen_values()
        attrs = range(0, self.values.shape[1] - 1)
        self.update_seed()
        errors = []
        for exclude_attr in attrs:
            use_attrs = attrs[:exclude_attr] + attrs[exclude_attr + 1:]
            getter = [slice(None), use_attrs]
            est = self.init_estimator()
            np.random.seed(self.seed)
            error = self.run_estimator(
                est, gen_values.X_train[getter], gen_values.y_train,
                gen_values.X_test[getter], gen_values.y_test
            )
            errors.append(error)
        errors = np.array(errors)
        if self.classification:
            ret_attrs = errors.argsort()
        else:
            ret_attrs = errors.argsort()[-1::-1]
        if return_scores:
            sign = 1 if self.classification else -1
            return errors.argsort()[-1::-1], sorted(errors, key=lambda x: sign * x)
        return ret_attrs

    @classmethod
    def get_important_from_file(cls, return_scores=False):
        import pickle
        import inspect
        import os
        class_file = inspect.getfile(cls)
        class_dir = os.path.dirname(class_file)
        f_path = os.path.join(class_dir, '../data/important.pickle')
        errors_dict = pickle.load(open(f_path))
        errors = errors_dict[cls.estimator.__name__]

        sign = 1 if cls.classification else -1
        attrs, errors = np.array(sorted(
            errors.items(), key=lambda x: x[1] * sign)).T
        attrs = attrs.astype(int)
        if return_scores:
            return attrs, errors
        return attrs


class MissingComparisonClassificationRunner(MissingComparisonRegressionRunner):
    classification = True
    estimator = tree.DecisionTreeClassifier


class ImputationComparisonRunner(BaseRunner):
    n_attr_missing = 1
    estimator = DecisionTreeClassifier
    base_name = "Base Name"
    test_split = 0.4

    def __init__(self, categorical_features=None, *args, **kwargs):
        super(ImputationComparisonRunner, self).__init__(*args, **kwargs)
        self.categorical_features = categorical_features or []
        self.update_seed()

    def update_seed(self):
        self._seed = np.random.randint(0, 1000)

    @staticmethod
    def calc_score(estimator_instance, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = estimator_instance.predict(X)
        return accuracy_score(y_pred, y)

    def run_once(self, estimator, percent_missing):
        from sklearn.feature_selection import SelectKBest, chi2
        from sklearn.cross_validation import train_test_split
        from sklearn.preprocessing import Imputer

        np.random.seed(self._seed)
        arr = self.values.copy()
        X, y = arr[:, :-1], arr[:, -1]

        best = SelectKBest(chi2, 'all')
        best.fit(X, y)
        best_args = best.scores_.argsort()[-1: -self.n_attr_missing - 1: -1]

        np.random.seed(self._seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_split)
        indexes = np.arange(X_train.shape[0])
        np.random.seed(self._seed)
        indexes_to_none = np.random.choice(indexes, int(len(indexes) * percent_missing), False)

        X_train_missing = X_train.copy()
        X_train_missing[indexes_to_none[:, None], best_args] = None

        imp_mean = Imputer()
        imp_mode = Imputer(strategy='most_frequent')
        for i in range(X_train_missing.shape[1]):
            if i not in best_args:
                continue
            if i in self.categorical_features:
                X_train_missing[:, [i]] = imp_mode.fit_transform(X_train_missing[:, [i]])
            else:
                X_train_missing[:, [i]] = imp_mean.fit_transform(X_train_missing[:, [i]])

        np.random.seed(self._seed)
        est = estimator()
        est.fit(X_train, y_train)
        score1 = self.calc_score(est, X_test, y_test)

        np.random.seed(self._seed)
        est = estimator()
        est.fit(X_train_missing, y_train)
        score2 = self.calc_score(est, X_test, y_test)

        X_train_without_missing = np.delete(X_train, indexes_to_none, 0)
        y_train_without_missing = np.delete(y_train, indexes_to_none, 0)
        np.random.seed(self._seed)
        est = estimator()
        est.fit(X_train_without_missing, y_train_without_missing)
        score3 = self.calc_score(est, X_test, y_test)

        return [score1, score2, score3]

    def run(self):
        results = []
        for i in np.concatenate((np.arange(0.1, 1, 0.1), np.arange(0.9, 0.99, 0.01))):
            results.append([i] + self.run_once(self.estimator, i))
        return np.array(results)

    def edit_ax(self, ax):
        ax.legend(loc='lower left')
        ax.set_ylim(0, 1)
        ax.set_title("{} - {} - {}".format(self.base_name, self.estimator.__name__, self.n_attr_missing))

    def plot_result(self, save_fig=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        names = ["", u"Completo", "Imputado", "Retirado"]
        result = np.array(self.run())
        for i in range(1, 4):
            ax.plot(result.T[0], result.T[i], label=names[i])
        self.edit_ax(ax)
        if not save_fig:
            plt.show()
        else:
            plt.savefig(save_fig)


class BaseMultipleRunner(object):
    runner = None

    def __init__(self, runner=None):
        self.runner = runner or self.runner
        self.runner = self.runner()

    def get_iter(self):
        raise NotImplementedError()

    def get_image_file(self, item):
        raise NotImplementedError()

    def prepare_runner(self, item):
        raise NotImplementedError()

    @classmethod
    def get_class_file(cls):
        import inspect
        import os
        return os.path.abspath(inspect.getfile(cls))

    def run(self):
        iter_ = self.get_iter()
        for item in iter_:
            self.prepare_runner(item)
            self.runner.plot_result(self.get_image_file(item))


class MissingComparisonMulipleStudy(BaseMultipleRunner):
    save_dir = None

    def __init__(self, runner=None):
        self.runner = self.runner or runner
        self.sub_folder = None
        assert issubclass(self.runner, MissingComparisonRegressionRunner)

    def get_estimators(self):
        return CLASSIFIER_ESTIMATORS if self.runner.classification else REGRESSION_ESTIMATORS

    def get_iter(self):
        from itertools import product
        r = self.runner([])
        best_args = r.get_best_args()
        attrs_list = [best_args[:i] for i in range(1, len(best_args))]
        estimators = self.get_estimators()
        return product(attrs_list, estimators)

    def get_save_dir(self):
        import os
        if not self.save_dir:
            self.save_dir = os.path.join(os.path.dirname(self.get_class_file()), '../')
        return self.save_dir

    def prepare_runner(self, item):
        attrs, estimator = item
        instance = self.runner(attrs)
        instance.estimator = estimator
        return instance

    def get_suffix(self, item):
        attrs, estimator = item
        attrs = '_'.join(str(x) for x in attrs)
        return "{}_{}".format(attrs, estimator.__name__)

    def get_file(self, folder, ext, item):
        import os
        full_folder = os.path.join(self.get_save_dir(), folder, self.get_sub_folder())
        if not os.path.isdir(full_folder):
            os.makedirs(full_folder)
        return os.path.join(full_folder, "{}.{}".format(self.get_suffix(item), ext))

    def get_sub_folder(self):
        from datetime import datetime

        if self.sub_folder is None:
            self.sub_folder = 'missing_best_args/{:%Y%m%d%H%M%S}'.format(datetime.now())
        return self.sub_folder

    def run(self, times=100):
        for item in self.get_iter():
            print item
            data_path = self.get_file('data/', 'pickle', item)
            img_path = self.get_file('imgs/', 'png', item)
            runner = self.prepare_runner(item)
            me = runner.get_multiple_est_error_values(times)
            save_data(tuple(me), data_path)
            fig, axes = plt.subplots(1, 2, sharey=True)
            runner.plot_multiple_est_error_values(me, axes)
            fig.savefig(img_path)
            plt.close(fig)


class MissingComparisonMuliplePercentStudy(MissingComparisonMulipleStudy):
    n_attr_missing = 3
    estimator = tree.DecisionTreeRegressor

    def prepare_runner(self, percent_missing):
        r = self.runner([])
        best_args = r.get_best_args()
        instance = self.runner(best_args[:3])
        instance.percent_missing = percent_missing
        instance.estimator = tree.DecisionTreeRegressor
        instance.seed = 57
        return instance

    def get_suffix(self, item):
        return str(item)

    def get_iter(self):
        return np.arange(0.1, 0.7, 0.1)

    @classmethod
    def plot(cls, dir):
        import glob
        import os
        import pickle
        pattern = os.path.join(dir, '*.pickle')
        biased_data = []
        random_data = []
        for fname in glob.glob(pattern):
            f = open(fname)
            data = pickle.load(f)
            biased, random = data
            biased_data.append(biased.mean(0))
            random_data.append(random.mean(0))

        x = np.arange(0.1, 0.7, 0.1)
        fig, axes = plt.subplots(1, 2)
        legends = ["Original", "Vizinhos", u"Média", "Retirado"]
        for ax, data in zip(axes, (biased_data, random_data)):
            plot_data = np.array(data).T
            for line, legend in zip(plot_data, legends):
                ax.plot(x, line, label=legend)
            ax.legend()
        plt.show()


class MissingComparisonMulipleAttrsStudy(MissingComparisonMulipleStudy):
    def get_iter(self):
        r = self.runner([])
        best_args = r.get_best_args()
        return [best_args[:i] for i in range(1, len(best_args))]

    def prepare_runner(self, item):
        instance = self.runner(item)
        return instance

    def plot(self, errors):
        colors = ['blue', 'green', 'red', 'yellow']
        fig, ax = plt.subplots(1, 1)
        errors = np.array(errors).T
        n_attrs = range(1, errors.shape[1] + 1)
        for error, color in zip(errors, colors):
            ax.plot(n_attrs, error, color=color)
        plt.show()

    def run(self):
        data = []
        for item in self.get_iter():
            print item
            runner = self.prepare_runner(item)
            runner.seed = 8798
            np.random.seed(runner.seed)
            gen_values = runner.get_gen_values()
            av = runner.get_all_values(gen_values)
            data.append(av)
        self.plot([x.random.error_values for x in data])


class MultipleNAttrStudy(BaseMultipleRunner):
    def get_iter(self):
        max_attrs = self.runner.values.shape[1] - 1
        return range(1, max_attrs)

    def prepare_runner(self, item):
        self.runner.n_attr_missing = item

    def get_image_file(self, item):
        import os
        class_file = self.get_class_file()
        return os.path.join(os.path.dirname(class_file), '../imgs/n_attrs_missing/{}_{:02d}.png'.format(self.runner.estimator.__name__, item))


class MultipleEstimatorsStudy(BaseMultipleRunner):

    def get_iter(self):
        estimators = [ensemble.RandomForestClassifier, ensemble.ExtraTreesClassifier, tree.DecisionTreeClassifier, tree.ExtraTreeClassifier, naive_bayes.GaussianNB]
        return estimators

    def prepare_runner(self, item):
        self.runner.estimator = item
        self.runner._seed = 100

    def get_image_file(self, item):
        import os
        class_file = self.get_class_file()
        return os.path.join(os.path.dirname(class_file), '../imgs/estimators/{}.png'.format(self.runner.estimator.__name__, item))
