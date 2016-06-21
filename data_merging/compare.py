import numpy as np


class Compare(object):
    def __init__(self, dms, lines=''):
        super(Compare, self).__init__()
        self.dms = dms
        self.dss = dms.get_ds()
        self.estimators = self.dss.get_estimators()
        self.lines = lines

    def compare(self, results):
        dss = self.dss
        ret = []
        for estimator, my_attr, forward_attr in results:
            np.random.seed(dss.seed)
            estimator.fit(dss.loader.data.iloc[:, my_attr], dss.loader.data.iloc[:, -1])
            np.random.seed(dss.seed)
            my_result = estimator.predict(dss.loader.test_data.iloc[:, my_attr])
            np.random.seed(dss.seed)
            my_error = dss.error(my_result, dss.loader.test_data.iloc[:, -1])

            np.random.seed(dss.seed)
            estimator.fit(dss.loader.data.iloc[:, forward_attr], dss.loader.data.iloc[:, -1])
            np.random.seed(dss.seed)
            forward_result = estimator.predict(dss.loader.test_data.iloc[:, forward_attr])
            np.random.seed(dss.seed)
            forward_error = dss.error(forward_result, dss.loader.test_data.iloc[:, -1])

            ret.append((estimator, my_attr, my_error, forward_attr, forward_error))
        return ret

    @staticmethod
    def get_my_attrs(lines):
        return [eval(l.split(';')[1]) for l in lines]

    def get_forward_attrs(self):
        attrs = []
        for e in self.estimators:
            attrs.append(self.dms.select_feature_forward(e))
        return attrs

    def build_result(self, my_attrs, forward_attrs):
        return [(e, my_attr, forward_attr) for (e, my_attr, forward_attr) in zip(self.estimators, my_attrs, forward_attrs)]

    def run(self):
        my_attrs = self.get_my_attrs(self.lines)
        forward_attrs = self.get_forward_attrs()
        result = self.build_result(my_attrs, forward_attrs)
        return self.compare(result)

    @staticmethod
    def print_result(result):
        print 'estimator;my_attrs;error_out;forward_attrs;error_out'
        for r in result:
            print '{};{};{};{};{}'.format(
                r[0].__class__.__name__,
                *r[1:]
            )
