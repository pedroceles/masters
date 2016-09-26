# -*- coding: utf-8 -*-
def get_df_profile(db_type='regression', treatment='neigh'):
    from glob import glob
    from imputation import df_transformer
    fs = glob(
        '/Users/pedroceles/dev/mestrado/dbs/{}/*/data/percent.pickle'.format(
            db_type
        ))
    df = df_transformer.DFAggregator(
        df_transformer.PercentDFTransformer, fs, div=False).get_df()
    df = df.loc[:, (slice(None), treatment, 'mean')]
    biased, random = df.values.T
    diff = abs(random - biased) / random
    bgt = biased > random
    df['diff'] = diff
    df['bgt'] = bgt
    return df, diff, bgt


def get_score_table(db_type):
    from glob import glob
    import pickle
    import pandas as pd
    fs = glob(
        '/Users/pedroceles/dev/mestrado/dbs/{}/*/data/important.pickle'.format(
            db_type
        ))
    table = []
    sort_value = 1 if db_type == 'classification' else -1
    error = "Ac." if db_type == 'classification' else "MAPE"
    for fname in fs:
        data_dict = pickle.load(open(fname))
        db_name = fname.split('/')[-3]
        for k, v in data_dict.items():
            if k.startswith('SGD'):
                continue
            atrr_values = sorted(v.items(), key=lambda x: sort_value * x[1])
            string = ", ".join(
                ["{} ({:.2f})".format(x, y * 100) for x, y in atrr_values])
            table.append([db_name, k, string])
    return pd.DataFrame(
        table, columns=["Database", "Estimator", "Attribute ({})".format(
            error
            )
        ]
    )
