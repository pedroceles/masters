from matrix import study
import pickle
import os
# mss = study.MatrixSingleStudy('/home/pedroceles/Dropbox/Mestrado/Projeto/pesquisas/bike/data/treated.csv')
# mss.loader.hide_data_independent(*range(4), percent=0.3)
# est = mss.get_estimators()[0]
# mss.calc_matrix(est)
# pickle.dump(mss, open(os.path.join(os.path.dirname(__file__), 'matrix.pickle'), 'wb'))
mss = pickle.load(open(os.path.join(os.path.dirname(__file__), 'matrix.pickle')))
mss.fill_data()
