# -*- coding: utf-8 -*-
def all_combinations(list_):
    from itertools import combinations
    result = []
    for i in range(len(list_)):
        result.extend(combinations(list_, i + 1))
    return result


def make_padrao(dss, dir_, estimators, v=0):
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
        for est in estimators:
            est_name = est.__class__.__name__.lower()
            fname = os.path.join(dir_, 'ouro_' + est_name + '.txt')
            pool.add_process(calc_score, args=(fname, dss, df, est, attrs))
    pool.start()


def get_errors_without(attrs, df):
    cols = []
    attrs = eval(attrs)
    for j, (attrs_, error) in enumerate(df.values):
        attrs_tuple = eval(attrs_)
        cont = False
        for i in attrs:
            if i in attrs_tuple:
                cont = True
                break
        if cont:
            continue
        cols.append((j, attrs_, error))
    return cols


def print_res(dfs):
    cols = ['regressor', 'attrs', 'min', 'max', 'no attrs', 'min', 'pos']
    print ';'.join(cols)
    for df, name in dfs:
        s = df.sort('error')
        top_attrs = s.values[0][0]
        cols = get_errors_without(top_attrs, s)
        pos, no_attrs, error = cols[0]
        rows = [name, top_attrs, s.values[0][1], s.values[-1][1], no_attrs, error, pos]
        rows = [str(x) for x in rows]
        print ';'.join(rows)


def calc_score(fname, dss, df, est, attrs):
    from lockfile import LockFile
    lock = LockFile(fname)
    file_ = open(fname, 'a')
    result = abs(dss.calc_score(est, df, attrs, use_seed=True).mean())
    lock.acquire()
    file_.write('{}; {}\n'.format(repr(attrs), result))
    lock.release()


def get_dfs(dir_):
    import pandas as pd
    import os
    dfs = []
    for fname in os.listdir(dir_):
        df = pd.read_csv(os.path.join(dir_, fname), ';', names=['a', 'error'])
        dfs.append((df, fname))
    return dfs


def get_tops(dir_):
    dfs = get_dfs(dir_)
    tops = []
    for df, fname in dfs:
        df = df.sort('error')
        top_attrs = eval(df.values[0][0])
        tops.append((fname, top_attrs))
    return tops
