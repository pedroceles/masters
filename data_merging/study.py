# -*- coding: utf-8 -*-
import pandas #noqa
import numpy as np
from data_merging.loader import DataLoader
from ml_metrics.elementwise import rmse


class DataSingleStudy(object):
    '''Class que abstrai o data loader em vem com funções para
    estudos da base de dados'''

    def __init__(self, fname, sep=',', hidden_partition=None, cv=2, error=rmse):
        '''
        hidden_partition: percentual de dados que serão escondidos
        cv: quantidade de folds para calculo do erro
        error: erro a ser calculado
        '''
        self.hidden_partition = hidden_partition
        self.cv = cv
        self.error = error
        self.fname = fname
        self.sep = sep
        self.get_loader()

    def get_loader(self, norm=True):
        self.loader = DataLoader(self.fname, norm=norm, sep=self.sep)

    @property
    def n_attr(self):
        return self.loader.data.shape[1] - 1

    def hide_data(self, *args, **kwargs):
        kwargs['hidden_partition'] = self.hidden_partition
        self.loader.hide_data(*args, **kwargs)

    def fill_data(self, backbones, n=3):
        self.loader.fill_data(backbones, n=n)

    def get_attr_error(self, cols):
        '''Calcula o erro das colunas entre os dados originais e os que foram previsto por fill_data'''
        from ml_metrics.elementwise import rmse
        dl = self.loader
        errors = {}
        for col in cols:
            data_to_calc = dl.data[dl.data.ix[:, col] != dl.hided_data.ix[:, col]]
            print data_to_calc.shape
            errors[col] = rmse(dl.data.ix[:, col], dl.hided_data.ix[:, col])
        return errors

    def run_once(self, cols1=[0, 1, 2, 6], cols2=[3, 4, 5, 7]):
        '''Esconde os dados de cols1, cols2 na proporção dada por hidden_partition
        retorna os erros de previsão do fill_data para cada coluna em cols1 e cols2'''
        self.hide_data(*[cols1, cols2])
        backbones = set(range(12)).difference(cols1 + cols2)
        self.fill_data(backbones)
        ret = self.get_attr_error(cols1 + cols2)
        return ret

    def calc_score(self, estimator, df, use_attrs=None):
        '''Calcular o erro de um estimador, dado um data_frame. considera-se que
        todas as colunas do df são usadas na previsão da última.'''
        from sklearn.metrics import make_scorer
        from sklearn.cross_validation import cross_val_score
        scorer = make_scorer(self.error, greater_is_better=False)
        target = df.values[:, -1]
        if not use_attrs:
            vals = df.values[:, :-1]
        else:
            vals = df.values[:, use_attrs]

        return cross_val_score(estimator, vals, target, scorer, cv=self.cv, n_jobs=1)

    def compare_estimators(self, estimators, backbones=[3, 4, 5, 6, 7]):
        ''' Dado os backbones, compara diversos estimadores
        Retorna um dicionario, onde a key é o estimador e o value é o erro
        da base de dados completa, o erro da base de dados com os dados preenchidos por
        fill_data e o delta percentual desses erros
        '''
        result = {}
        # total de colunas, a última sendo o target
        total_attr = range(self.n_attr)
        hide_cols = list(set(total_attr).difference(backbones))
        # dividindo as colunas em duas
        hide_cols = [hide_cols[:len(hide_cols) / 2], hide_cols[len(hide_cols) / 2:]]
        self.hide_data(*hide_cols)
        self.fill_data(backbones, n=3)
        for est in estimators:
            error_data = abs(self.calc_score(est, self.loader.data)).mean()
            error_filled = abs(self.calc_score(est, self.loader.hided_data)).mean()
            delta_percent = (error_filled - error_data) / error_data
            result[est] = error_data, error_filled, delta_percent
        return result

    @staticmethod
    def get_estimators():
        from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.linear_model import SGDRegressor
        from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
        ests = [RandomForestRegressor, ExtraTreeRegressor, AdaBoostRegressor, GradientBoostingRegressor, SVR, SGDRegressor, DecisionTreeRegressor]
        return [e() for e in ests]

    def make_comparison(self, backbones=[3, 4, 5, 6, 7]):
        '''classe que pega os estimadores de get_estimators e retorna os valores
        em de compare_estimators de forma matricial'''
        np.random.seed()
        ests = self.get_estimators()
        errors = self.compare_estimators(ests, backbones)
        error_arrays = np.array([errors[est] for est in ests]).T
        return error_arrays

    def make_multi_regression(self, cols):
        '''Faz a regressão, usando SVR, e utilizando apenas colunas específicas'''
        from sklearn.metrics import make_scorer
        from sklearn.cross_validation import cross_val_score
        from sklearn.svm import SVR
        errors = {}
        estimator = SVR()
        df = self.loader.data
        for _cols in cols:
            key = frozenset(set(range(self.n_attr)).difference(_cols))
            scorer = make_scorer(self.error, greater_is_better=False)
            vals = df.values[:, _cols]
            target = df.values[:, -1]
            errors[key] = abs(cross_val_score(estimator, vals, target, scorer, cv=self.cv, n_jobs=-1)).mean()
        return errors


class DataMultiStudy(object):
    '''Wrapper para fazer estudos que requerem chamadas de diferentes DataSingleStudy'''

    def __init__(self, fname, sep=',', hidden_partition=None, cv=2, error=rmse):
        self.hidden_partition = hidden_partition
        self.cv = cv
        self.error = error
        self.fname = fname
        self.sep = sep

    def get_ds(self):
        '''retorna uma instancia de DataSingleStudy'''
        self.last_ds = DataSingleStudy(
            self.fname,
            self.sep,
            hidden_partition=self.hidden_partition,
            cv=self.cv,
            error=self.error
        )
        return self.last_ds

    @property
    def ds_n_attr(self):
        ds = getattr(self, 'last_ds', None) or self.get_ds()
        return ds.n_attr

    def check_error_growth(self, col=0):
        '''Mostra a evolução dos erros do fill data para a coluna 'col'
        a medida que outras colunas vão sendo escondidas também, e menor é o número
        de backbones'''
        cols1 = [col]
        cols2 = []
        errors_col = []
        for i in range(0, self.n_attr):
            if i == col:
                continue
            cols2.append(i)
            DS = self.get_ds()
            errors_all = DS.run_once(cols1, cols2)
            errors_col.append(errors_all[col])
        return errors_col

    def check_error_growth_thread(self, col=0):
        '''Mostra a evolução dos erros do fill data para a coluna 'col'
        a medida que outras colunas vão sendo escondidas também, e menor é o número
        de backbones. Utiliza múltiplos processos'''
        from multiprocessing import Manager
        from copy import copy
        cols1 = [col]
        cols2 = []

        pool = MyPool(verbosity=1)
        manager = Manager()
        errors_col = manager.list()

        def run(cols1, cols2, i):
            DS = self.get_ds()
            errors_all = DS.run_once(cols1, cols2)
            errors_col.append((errors_all[col], i))
        for i in range(0, self.ds_n_attr):
            if i == col:
                continue
            cols2.append(i)
            pool.add_process(run, args=(copy(cols1), copy(cols2), i))
        pool.start()
        errors_col = list(errors_col)
        errors_col = sorted(errors_col, key=lambda x: x[1])
        return [x[0] for x in errors_col]

    def check_error_growth_all_variables(self):
        import matplotlib.pyplot as plt
        errors = []
        range_x = []
        for i in range(0, self.ds_n_attr):
            errors_col = self.check_error_growth_thread(i)
            errors.extend(errors_col)
            range_x.extend([i + float(x)/30 for x in range(len(errors_col))])
        plt.scatter(range_x, errors)
        plt.show()

    def check_error_growth_all_variables_thread(self):
        from multiprocessing import Manager
        manager = Manager()
        errors = manager.dict()
        pool = MyPool(verbosity=1)

        def run(i):
            print "running", i
            errors_col = self.check_error_growth_thread(i)
            print "end", i
            errors[i] = errors_col

        for i in range(self.ds_n_attr):
            pool.add_process(run, args=(i,))
        pool.start()
        errors = dict(errors)
        return errors

    def make_multi_comparison(self, backbones=[3, 4, 5, 6, 7], times=4):
        '''Execta a comparação de estimadores várias vezes em multiplos processos
        e retorna a média'''
        import multiprocessing as mp
        pool = MyPool(verbosity=1)
        manager = mp.Manager()
        errors = manager.list()

        def run():
            DS = self.get_ds()
            error_arrays = DS.make_comparison(backbones)
            errors.append(error_arrays)
        self._run = run
        for i in range(times):
            pool.add_process(run)
        pool.start()
        return np.array(errors).mean(axis=0)

    def make_multi_backbones_comparison(self, backbones_array=[], times=4):
        '''Faz a comparação de estimadores para vários backbones'''
        max_deltas = {}

        def run(backbones, i):
            print "Started {}. Backbones{}".format(i, backbones)
            error_arrays = self.make_multi_comparison(backbones, times=times)
            max_delta = max(error_arrays[2])
            max_deltas[tuple(sorted(backbones))] = max_delta
            print "Finished {}. Backbones{}".format(i, backbones)

        for counter, backbones in enumerate(backbones_array):
            run(backbones, counter)

        return max_deltas

    def select_most_important(self, times=4):
        from itertools import combinations
        combs = combinations(range(self.ds_n_attr), self.ds_n_attr - 1)
        result = self.make_multi_backbones_comparison(combs, times=times)  # TODO Pedro Celes - [13-05-2015]: ver times
        max_deltas = {}
        for k, v in result.items():
            new_key = set(range(self.ds_n_attr)).difference(k)
            max_deltas[new_key.pop()] = v
        return sorted(max_deltas.items(), key=lambda x: -x[1])

    def plot_most_important(self, save=False, times=4, **kwargs):
        import matplotlib.pyplot as plt
        important = self.select_most_important(times=times)
        important = [x[0] for x in important]
        # TODO Pedro Celes - [13-05-2015]: ver times
        deltas_most = self.make_multi_backbones_comparison(backbones_array=[important[0: i] for i in range(1, self.ds_n_attr)], times=times)
        deltas_most = [z[1] for z in sorted(deltas_most.items(), key=lambda x: len(x[0]))]
        deltas_less = self.make_multi_backbones_comparison(backbones_array=[important[-1: -i: -1] for i in range(2, self.ds_n_attr + 1)], times=times)
        deltas_less = [z[1] for z in sorted(deltas_less.items(), key=lambda x: len(x[0]))]
        plt.plot(range(1, self.ds_n_attr), deltas_most, label="--->")
        plt.plot(range(1, self.ds_n_attr), deltas_less, label="<---")
        plt.title("{}".format(important))
        plt.legend()
        if not save:
            plt.show()
        else:
            plt.savefig(kwargs['fname'])
            plt.clf()

    def check_important_multi_hidden(self, times=4):
        hiddens = np.arange(0, 0.9, 0.1)
        result = []
        for hidden in hiddens:
            print 'hidden', hidden
            self.hidden_partition = hidden
            important_array = self.select_most_important(times=times)
            result.append(important_array)
        return result

    def plot_comparison(self, backbones=[3, 4, 5, 6, 7], fn_kwargs={}):
        import matplotlib.pyplot as plt

        error_arrays = self.make_multi_comparison(backbones, **fn_kwargs)
        ests = DataSingleStudy.get_estimators()

        def _format_text(text, width=15):
            '''Transforma Camel case em space_separeted
            e limita o text a ter width'''
            import re
            import textwrap
            text = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", text)
            return textwrap.fill(text, width, break_long_words=False)

        bar_width = 0.2
        ax = plt.subplot()
        x_pos = np.arange(error_arrays.shape[1]) + 1
        ax.bar(x_pos - bar_width, error_arrays[0] * 100, bar_width, label="Erro Real")
        ax.bar(x_pos, error_arrays[1] * 100, bar_width, color="red", label=u"Erro Pós")
        ax.legend()

        ax.set_xlim((0, 8))
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels([_format_text(x.__class__.__name__) for x in ests])
        ax.set_title("Backbones {}".format(backbones))

        ax2 = ax.twinx()
        ax2.plot(x_pos, 100 * error_arrays[2], label='delta', color='k')
        max_delta = error_arrays[2].max()
        ax2.set_ylim((-50, max_delta * 100 * 1.2))
        ax2.set_yticks(range(0, 21, 5))
        ax2.set_yticklabels(range(0, 21, 5))
        index_max_delta = error_arrays[2].argmax()
        ax2.annotate(
            u"Máx: {:.2f}".format(max_delta * 100),
            (index_max_delta + 1, max_delta * 100),
            (index_max_delta + 1, max_delta * 100 + 3),
            arrowprops=dict(arrowstyle='->',
                            connectionstyle="arc,angleA=0,armA=30,rad=10"
                            ),
        )
        plt.show()

    def save_multi_comparison_plot(self, folder, times):
        from copy import copy
        import os
        file_name = 'hidden{}.png'
        for i in np.arange(0.1, 1, 0.1):
            dms = copy(self)
            dms.hidden_partition = i
            dms.plot_most_important(True, times=times, fname=os.path.join(
                folder, file_name.format(i)
            ))
#
#
# def make_regression(self, df, cols):
#     from sklearn.metrics import make_scorer
#     from sklearn.cross_validation import cross_val_score
#     from sklearn.svm import SVR
#     estimator = SVR()
#     scorer = make_scorer(rmse, greater_is_better=False)
#     vals = df.values[:, cols]
#     target = df.values[:, -1]
#     return abs(cross_val_score(estimator, vals, target, scorer, cv=2, n_jobs=-1)).mean()
#
#
# def make_all_regressions(self, df):
#     from itertools import chain, combinations
#     min_error = 1
#     min_col = []
#     combs = chain(*[combinations(range(11), i) for i in range(1, 12)])
#     for cols in combs:
#         error = make_regression(df, cols)
#         min_error = min(error, min_error)
#         if min_error == error:
#             print cols, min_error
#             min_col = cols
#     return min_col, error

import multiprocessing as mp


class MyPool(object):
    def __init__(self, processes=None, verbosity=0):
        self.processes = processes or mp.cpu_count()
        self.queue = []
        self.current_processes = []
        self.has_started = False
        self.verbosity = verbosity

    def add_process(self, fun, args=(), kwargs={}):
        if self.has_started:
            raise ValueError("Pool has started")
        process = mp.Process(target=fun, args=args, kwargs=kwargs)
        self.queue.append(process)

    def start(self):
        self.has_started = True
        while self.queue:
            self.current_processes = self.queue[:self.processes]
            if self.verbosity >= 1:
                print "Starting batch of {} processes".format(len(self.current_processes))
            self.queue[:self.processes] = []
            for i in self.current_processes:
                i.start()
            for i in self.current_processes:
                i.join()
