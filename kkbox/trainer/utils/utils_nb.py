import pandas as pd, seaborn as sns, re


from matplotlib import pyplot as plt

# def flat(data, multi_col):
#     """Flat the splitted string multivariate feature with the target
#         - catg1|catg2, 1
#         - catg2|catg3, 0
#                to
#         -   catg1, 1
#         -   catg2, 1
#         -   catg2, 0
#         -   catg3, 0
#     """
#     msno_ary, multi_list, target = [], [], []
#     lens = data[multi_col].map(lambda e: len(e) if type(e) in (list, tuple) else 1)
#     data[multi_col].map(lambda e: multi_list.extend(e) if type(e) in (list, tuple) else multi_list.append(None))
#     def map_fn(tp):
#         msno_ary.extend([tp[1]] * tp[0])
#         target.extend([tp[2]] * tp[0])
#
#     pd.Series(list(zip( lens, data.msno, data.target ))).map(map_fn)
#     return msno_ary, multi_list, target

def flatten(data, uni_cols:list, m_col, target):
    # sep = '|'
    multi = data[m_col].copy()
    multi.loc[multi.isna()] = multi[multi.isna()].map(lambda e: ('',))
    series = pd.Series(list(zip(list(data[uni_cols].values),
                                list(multi), #  .str.split(f'\s*{re.escape(sep)}\s*')),
                                list(data[target]))))

    def map_fn(e):
        uni, mul, label = e
        uni = tuple(uni)
        cache.extend([uni + (ele, label) for ele in mul])

    columns = uni_cols + [m_col, target]
    cache = []
    series.map(map_fn)
    return pd.DataFrame(columns=columns, data=cache)


def univ_boxplot(df, colname):
    """Draw boxplot to observe the statistic values"""
    anchor_grp = df.groupby(colname)
    agg = pd.DataFrame({'mean': anchor_grp.target.mean().drop(''),
                        'sum': anchor_grp.target.sum().drop('')})
    agg['popular'] = agg['mean'] * agg['sum']
    plt.title(f'{colname} Popular Distribution')
    ax = sns.boxplot(x=agg.popular)
    plt.show()


def heatmap(data, *cols, xtick=None, ytick=None, annot=True, fmt='.2f'):
    f, axs = plt.subplots(1, 2, figsize=(16, 4))

    def draw(chop, axis):
        pivot_params = list(cols) + ['target']
        g = chop.groupby(list(cols)).target
        mean_ = g.mean().reset_index().pivot(*pivot_params)
        count_ = g.size().reset_index().pivot(*pivot_params)
        if xtick is not None or ytick is not None:
            mean_ = mean_.reindex(index=ytick, columns=xtick)
            count_ = count_.reindex(index=ytick, columns=xtick)

        sns.heatmap(mean_.fillna(0), annot=annot, ax=axis[0], fmt=fmt)
        sns.heatmap(count_.fillna(0), annot=annot, ax=axis[1], fmt=fmt)
        axis[0].set_title('mean')
        axis[1].set_title('count')

    draw(data, axs)
    plt.show()

# def heatmap(data, *cols, col=None, row=None, xtick=None, ytick=None, annot=True):
#     col_val, row_val = [0], [0]
#     if col is not None:
#         col_val = sorted(data[col].dropna().unique())
#
#     if row is not None:
#         row_val = sorted(data[row].dropna().unique())
#
#     f, axs = plt.subplots(len(row_val), len(col_val), figsize=(12 * len(col_val), 8 * len(row_val))) # figsize=(12, 4)
#
#     def draw(chop, axis):
#         pivot_params = list(cols) + ['target']
#         g = chop.groupby(list(cols)).target
#         mean_ = g.mean().reset_index().pivot(*pivot_params)
#         # count_ = g.size().reset_index().pivot(*pivot_params)
#         if xtick is not None or ytick is not None:
#             mean_ = mean_.reindex(index=ytick, columns=xtick)
#             # count_ = count_.reindex(index=ytick, columns=xtick)
#
#         sns.heatmap(mean_.fillna(0), annot=annot, ax=axis)
#         # sns.heatmap(count_.fillna(0), annot=annot, ax=axs[1])
#         axis.set_title('mean')
#         # axs[1].set_title('count')
#
#     if col is not None and row is not None:
#         for i, r in enumerate(row_val):
#             for j, c in enumerate(col_val):
#                 draw(data.query(f"{row} == '{r}' and {col} == '{c}'"), axs[i, j])
#     elif row is not None:
#         for i, r in enumerate(row_val):
#             draw(data.query(f"{row} == '{r}'"), axs[i])
#     elif col is not None:
#         for j, r in enumerate(col_val):
#             draw(data.query(f"{col} == '{r}'"), axs[j])
#     else:
#         draw(data, axs)
#
#     plt.show()

# # Because of performance of pandas.to_csv too slow, use this will be greate
# def to_csv(path, df, header=True):
#     with open(path, 'w', newline='') as fp:
#         w = csv.writer(fp)
#         if header:
#             w.writerows(df.columns[None, :])
#         w.writerows(df.values)

# n_msno = data.msno.nunique()
# cache = {'progress': 0}
# def map_msno(pipe):
#     def set_cols(col, weights):
#         try:
#             weights = weights.drop('')
#         except:
#             pass
#         if len(weights) > 0:
#             ret[f'msno_{col}_hist'] = tuple(weights.index)
#             ret[f'msno_{col}_count'] = tuple(weights['count'].values.astype(float))
#             ret[f'msno_{col}_mean'] = tuple(weights['mean'].values.astype(float))
#         else:
#             ret[f'msno_{col}_count'] = ('',)
#             ret[f'msno_{col}_mean'] = (0.,)
#             ret[f'msno_{col}_mean'] = (0.,)
#
#     cache['progress'] += 1
#     if cache['progress'] % 100 == 0:
#         print(f'\r {cache["progress"]}/{n_msno} processed.', end='')
#
#     ret = pd.Series()
#     # pos_pipe = pipe.query('target == 1')
#     len_ = len(pipe)
#     for col in ('source_system_tab', 'source_screen_name', 'source_type', 'language'):
#         weights = pipe.groupby(col).target.agg(['count', 'mean'])
#         # label_ary, weight_ary = [], []
#         # pd.Series(list(zip(pipe.msno, pipe[col], pipe.target))).map(leave_one_out)
#         set_cols(col, weights)
#
#     for col in ('genre_ids', 'artist_name', 'composer', 'lyricist'):
#         flated, target = utils_nb.flat(pipe, col)
#         weights = pd.DataFrame({col: flated, 'target': target}) \
#             .groupby(col).target.agg(['count', 'mean'])
#         # label_ary, weight_ary = [], []
#         # pd.Series(list(zip(pipe.msno, pipe[col], pipe.target))).map(leave_one_out)
#         set_cols(col, weights)
#
#     ret['msno_song_query'] = tuple(pos_pipe.song_id.unique()) if len(pos_pipe) else ('',)
#     ret['msno_song_query_weights'] = (1.,) * len(ret['msno_song_query'])
#     return ret


# def gen(inputs):
#     n_batch = 500
#     defaults = dict(zip(metadata.HEADER, metadata.HEADER_DEFAULTS))
#
#     # Multivariate columns can't use function fillna ...
#     def multi_fillna(df, colname):
#         na_value = tuple(defaults[colname])
#         na_conds = df[colname].isna().values
#         df.loc[na_conds, colname] = df[na_conds][colname].map(lambda na: na_value)
#
#     counter = 0
#     for _, data in inputs.groupby(np.arange(len(inputs)) // n_batch):
#         merge = data.merge(members, how='left', on='msno') \
#             .merge(songs, how='left', on='song_id')[metadata.HEADER]
#         # Join could make NaN columns, fill the default value
#         for colname in metadata.USER_FEATURES + metadata.SONG_FEATURES:
#             if colname.endswith('_hist') or colname.endswith('_count') or colname.endswith('_mean') or \
#                     colname in ('genre_ids', 'artist_name', 'composer', 'lyricist'):
#                 multi_fillna(merge, colname)
#             else:
#                 merge[colname] = merge[colname].fillna(defaults[colname][0])
#         counter += len(data)
#         print(f'{counter} processed ...')
#         # DataFrame.itertuple faster than DataFrame.iterrows
#         for row in merge.itertuples(index=False):
#             yield row