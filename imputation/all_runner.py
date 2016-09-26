# -*- coding: utf-8 -*-
from base_runner import (
    MissingComparisonClassificationRunner,
    MissingComparisonRegressionRunner
)
from importer import Importer
import numpy as np

import sklearn
from sklearn import neighbors
# from sklearn import svm
REGRESSION_ESTIMATORS = [
    sklearn.tree.DecisionTreeRegressor,
    sklearn.ensemble.RandomForestRegressor,
    sklearn.ensemble.AdaBoostRegressor,
    # linear_model.SGDRegressor,
    neighbors.KNeighborsRegressor,
    # sklearn.ensemble.GradientBoostingRegressor,
    # svm.LinearSVR
]

CLASSIFICATION_ESTIMATORS = [
    sklearn.tree.DecisionTreeClassifier,
    sklearn.ensemble.RandomForestClassifier,
    sklearn.ensemble.AdaBoostClassifier,
    # linear_model.SGDClassifier,
    neighbors.KNeighborsClassifier,
    # linear_model.LinearRegression,
    # sklearn.ensemble.GradientBoostingClassifier,
    # svm.LinearSVC
]


class RunnerImporter(Importer):
    def run(self, times):
        self.import_all_classes()
        for klass in self.classes:
            self.run_once(klass, times)

    def get_data_dir(self, klass):
        import inspect
        import os
        klass_dir = os.path.dirname(inspect.getfile(klass))
        data_dir = os.path.join(klass_dir, '../data/')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir


class ClassificationMixin(object):
    estimators = CLASSIFICATION_ESTIMATORS

    def __init__(self, *args, **kwargs):
        kwargs['base_class'] = MissingComparisonClassificationRunner
        kwargs['glob'] = \
            ('/Users/pedroceles/dev/mestrado/dbs/classification/*/src')
        super(ClassificationMixin, self).__init__(*args, **kwargs)


class RegressionMixin(object):
    estimators = REGRESSION_ESTIMATORS

    def __init__(self, *args, **kwargs):
        kwargs['base_class'] = MissingComparisonRegressionRunner
        kwargs['glob'] = \
            ('/Users/pedroceles/dev/mestrado/dbs/regression/*/src')
        super(RegressionMixin, self).__init__(*args, **kwargs)


class ImportanceMixing(object):
    def run_once(self, klass, times=100):
        from collections import defaultdict
        import os
        import pickle
        estimators_errors = {}
        for estimator in self.estimators:
            error_dict = defaultdict(list)
            for time in range(times):
                print time, estimator, klass
                instance = klass([])
                instance.estimator = estimator
                attrs_important, errors = instance.get_important(
                    return_scores=True
                )
                for attr, error in zip(attrs_important, errors):
                    error_dict[attr].append(error)
            means = {}
            for attr, error_list in error_dict.items():
                means[attr] = np.mean(error_list)
            estimators_errors[estimator.__name__] = means
            save_dir = self.get_data_dir(klass)
            save_file = os.path.join(save_dir, 'important.pickle')
            pickle.dump(estimators_errors, open(save_file, 'w'))


class PercentAttrMixing(object):
    def run_once(self, klass, times=100):
        import math
        import os
        import pickle
        percents = [0.05, 0.1, 0.25, 0.5, 0.75]
        data = {}
        for estimator in self.estimators:
            for percent in percents:
                print percent, klass, estimator
                klass.percent_missing = 0.5
                klass.estimator = estimator
                attrs = klass.get_important_from_file()
                n_attrs = len(attrs)
                n_attrs_to_use = math.ceil(n_attrs * percent)
                if n_attrs_to_use == n_attrs:
                    n_attrs_to_use -= 1
                instance = klass(attrs[0:n_attrs_to_use])
                biased, random = instance.mp_get_multiple_est_error_values(
                    times=times)
                data[(estimator.__name__, percent)] = {
                    'biased': biased,
                    'random': random,
                }
                save_dir = self.get_data_dir(klass)
                save_file = os.path.join(save_dir, 'percent.pickle')
                pickle.dump(data, open(save_file, 'w'))


class PercentColMixing(object):
    def run_once(self, klass, times=100):
        import os
        import pickle
        percents = [0.05, 0.1, 0.20, 0.25, 0.30, 0.4, 0.5, 0.6]
        data = {}
        estimator = self.estimators[0]
        for percent in percents:
            print percent, klass
            klass.estimator = estimator
            klass.percent_missing = percent
            attrs = klass.get_important_from_file()[:1]
            instance = klass(attrs)
            biased, random = instance.mp_get_multiple_est_error_values(
                times=times)
            data[(estimator.__name__, percent)] = {
                'biased': biased,
                'random': random,
            }
            save_dir = self.get_data_dir(klass)
            save_file = os.path.join(save_dir, 'percent_col.pickle')
            pickle.dump(data, open(save_file, 'w'))


class DBInfoMixin(object):
    def run_once(self, klass):
        result = []
        for estimator in self.estimators:
            klass.estimator = estimator
            instance = klass([])
            values = instance.values
            important, scores = instance.get_important_from_file(True)
            important_gap = scores[0] / scores[-1]
            name = klass.__name__.replace('MissingComparisonRunner', '')
            result.append([
                name,
                values.shape[0],
                values.shape[1],
                estimator.__name__,
                important_gap]
            )
        return result

    def run(self):
        import pandas as pd
        self.import_all_classes()
        data = []
        for klass in self.classes:
            data.extend(self.run_once(klass))
            df = pd.DataFrame(data)
            df.columns = ['class', 'n_instances', 'n_attrs', 'estimator', 'importance_gap']
        return df


class ImportanceRegressionImporter(
    RegressionMixin, ImportanceMixing, RunnerImporter
):
    pass


class ImportanceClassificationImporter(
    ClassificationMixin, ImportanceMixing, RunnerImporter
):
    pass


class PercentAttrRegressionImporter(
    RegressionMixin, PercentAttrMixing, RunnerImporter
):
    pass


class PercentAttrClassificationImporter(
    ClassificationMixin, PercentAttrMixing, RunnerImporter
):
    pass


class PercentColRegressionImporter(
    RegressionMixin, PercentColMixing, RunnerImporter
):
    pass


class PercentColClassificationImporter(
    ClassificationMixin, PercentColMixing, RunnerImporter
):
    pass


class DBInfoClassificationImporter(
    ClassificationMixin, DBInfoMixin, RunnerImporter
):
    pass


class DBInfoRegressionImporter(
    RegressionMixin, DBInfoMixin, RunnerImporter
):
    pass
