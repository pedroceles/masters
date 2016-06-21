# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, clone


class BaseReducedModelEstimator(BaseEstimator):
    def __init__(self, base_estimator_class, *args, **kwargs):
        self.base_estimator_class = base_estimator_class
        super(BaseReducedModelEstimator, self).__init__(*args, **kwargs)

    def attrs_except(self, attrs, attr_exclude):
        attrs = list(attrs)
        return attrs[:attr_exclude] + attrs[attr_exclude + 1:]

    def calc_score(self, y_true, y_pred):
        from base_runner import mape
        return mape(y_true, y_pred)

    def fit(self, X, y, categorical_features=None):
        pass

    def predict(self, X):
        X_copy = X.copy()
        self.impute(X_copy)

        result = np.empty((X_copy.shape[0], ))
        predictions = {}
        nans = np.isnan(X)
        attrs = np.arange(X_copy.shape[1])
        for i, row in enumerate(X_copy):
            nan = nans[i]
            attrs_nan = np.where(nan)[0]
            preds = np.empty(attrs_nan.shape[0])
            for j, attr in enumerate(attrs_nan):
                attrs_to_train = self.attrs_except(attrs, attr)
                pred = predictions.setdefault(
                    attr, self.estimators[attr]['estimator'].predict(
                        X_copy[:, attrs_to_train]))[i]
                preds[j] = pred
            if not attrs_nan.shape[0]:
                pred = predictions.setdefault(
                    -1, self.estimators[-1].predict(
                        X_copy))[i]
                result[i] = pred
            else:
                result[i] = self.get_pred(preds, attrs_nan)
        return result

    def impute(self, X):
        from sklearn.preprocessing import Imputer
        imp_mean = Imputer()
        imp_mode = Imputer(strategy='most_frequent')
        for i in range(X.shape[1]):
            if i in self.categorical_features:
                X[:, [i]] = imp_mode.fit_transform(X[:, [i]])
            else:
                X[:, [i]] = imp_mean.fit_transform(X[:, [i]])

    def get_pred(self, preds, attrs_nan):
        raise NotImplementedError()


class OriginalReducedModelRegressor(BaseReducedModelEstimator):
    def est_object(self, estimator):
        return {
            'estimator': estimator,
            'score': 1,
        }

    def get_pred(self, preds, attrs_nan):
        return preds.mean()

    def fit(self, X, y, categorical_features=None):
        self.categorical_features = categorical_features or []
        base_est = self.base_estimator_class()
        self.estimators = []
        attrs = range(X.shape[1])
        for attr in attrs:
            attrs_to_train = self.attrs_except(attrs, attr)
            est = clone(base_est)
            est.fit(X[:, attrs_to_train], y)
            self.estimators.append(
                self.est_object(
                    est,
                )
            )
        self.estimators.append(base_est.fit(X, y))
        return self


class ReducedModelRegressor(BaseReducedModelEstimator):
    def get_weights(self, attrs_nan):
        scores = [self.estimators[attr]['score'] for attr in attrs_nan]
        return 1 / np.array(scores, float)

    def get_pred(self, preds, attrs_nan):
        weights = self.get_weights(attrs_nan)
        return (preds * weights).sum() / weights.sum()

    def est_object(self, estimator, y_true, y_pred):
        return {
            'estimator': estimator,
            'score': self.calc_score(y_true, y_pred)
        }

    def fit(self, X, y, categorical_features=None):
        self.categorical_features = categorical_features or []
        base_est = self.base_estimator_class()
        self.estimators = []
        attrs = range(X.shape[1])
        for attr in attrs:
            attrs_to_train = self.attrs_except(attrs, attr)
            est = clone(base_est)
            est.fit(X[:, attrs_to_train], y)
            self.estimators.append(
                self.est_object(
                    est, y, est.predict(X[:, attrs_to_train])
                )
            )
        self.estimators.append(base_est.fit(X, y))
        return self
