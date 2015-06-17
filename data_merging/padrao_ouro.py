# -*- coding: utf-8 -*-
def all_combinations(list_):
    from itertools import combinations
    result = []
    for i in range(len(list_)):
        result.extend(combinations(list_, i + 1))
    return result


def make_padrao(dss, dir_, v=0):
    '''
    dss: Instancia de DataSingleStudy,
    dir: diretório para onde serão salvos os arquivos com os dados

    Gera previsões para todas as combinações possíveis de atributos que serão utilizados.
    Para cada estimador em dss.get_estimators gera um arquivo, cada linha do arquivo
    cada linha representa uma combinaçõ diferente, onde há o erro associado a previsão
    utilizando a combinação da linha.
    '''
    import os
    from study import MyPool
    df = dss.loader.data
    total_attr = df.values.shape[1] - 1
    range_total_attr = range(total_attr)
    all_combs = all_combinations(range_total_attr)
    pool = MyPool(verbosity=v)
    for attrs in all_combs:
        # f = open(fname, 'w')
        for est in dss.get_estimators():
            est_name = est.__class__.__name__.lower()
            fname = os.path.join(dir_, 'ouro_' + est_name + '.txt')
            pool.add_process(calc_score, args=(fname, dss, df, est, attrs))
    pool.start()


def calc_score(fname, dss, df, est, attrs):
    from lockfile import LockFile
    lock = LockFile(fname)
    lock.acquire()
    file_ = open(fname, 'a')
    result = abs(dss.calc_score(est, df, attrs).mean())
    file_.write('{}; {}\n'.format(repr(attrs), result))
    lock.release()
