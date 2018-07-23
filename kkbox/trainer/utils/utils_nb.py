import pandas as pd, seaborn as sns

from matplotlib import pyplot as plt

def flat(data, multi_col):
    """Flat the splitted string multivariate feature with the target
        - catg1|catg2, 1
        - catg2|catg3, 0
               to
        -   catg1, 1
        -   catg2, 1
        -   catg2, 0
        -   catg3, 0
    """
    msno_ary, multi_list, target = [], [], []
    lens = data[multi_col].map(lambda e: len(e) if type(e) in (list, tuple) else 1)
    data[multi_col].map(lambda e: multi_list.extend(e) if type(e) in (list, tuple) else multi_list.append(None))
    def map_fn(tp):
        msno_ary.extend([tp[1]] * tp[0])
        target.extend([tp[2]] * tp[0])

    pd.Series(list(zip( lens, data.msno, data.target ))).map(map_fn)
    return msno_ary, multi_list, target


def univ_boxplot(df, colname):
    """Draw boxplot to observe the statistic values"""
    anchor_grp = df.groupby(colname)
    agg = pd.DataFrame({'mean': anchor_grp.target.mean().drop(''),
                        'sum': anchor_grp.target.sum().drop('')})
    agg['popular'] = agg['mean'] * agg['sum']
    plt.title(f'{colname} Popular Distribution')
    ax = sns.boxplot(x=agg.popular)
    plt.show()


def multi_catg_heatmap(data, multi_col, col):
    """Some features are category string, which contains maybe camma splitted,
     means multiple in a grid, called multivariate feature, so this function provide
     [univariate feature x multivariate feature] features heatmap with the mean of target
    """
    col_ary, multi_list_ary, target = [], [], []

    lens = data[multi_col].map(lambda e: len(e) if isinstance(e, tuple) else 1)
    data[multi_col].map(lambda e: multi_list_ary.extend(e) if isinstance(e, tuple) else multi_list_ary.append(None))
    pd.Series(list(zip(lens, data[col]))).map(lambda tp: col_ary.extend([tp[1]] * tp[0]))
    pd.Series(list(zip(lens, data.target))).map(lambda tp: target.extend([tp[1]] * tp[0]))

    df = pd.DataFrame({multi_col: multi_list_ary, col: col_ary, 'target': target})
    hm = df.groupby([multi_col, col]).target.mean().reset_index(name='target').pivot(multi_col, col, 'target').fillna(0)
    plt.figure(figsize=(10, 10))
    sns.heatmap(hm)
    plt.show()
    return hm


def heatmap(data, *cols, annot=True):
    f, axs = plt.subplots(1, 3, figsize=(20, 5))
    col1, col2 = cols[0], cols[1]
    grp = pd.DataFrame({col1: data[col1].values, col2: data[col2].values, 'target': data.target.values}).groupby(cols)

    pivot_params = list(cols) + ['target']
    g = data.groupby(cols).target
    mean_ = g.mean().reset_index().pivot(*pivot_params)
    count_ = g.size().reset_index().pivot(*pivot_params)
    sum_ = g.sum().reset_index().pivot(*pivot_params)

    sns.heatmap(mean_.fillna(0), annot=annot, ax=axs[0])
    sns.heatmap(count_.fillna(0), annot=annot, ax=axs[1])
    sns.heatmap(sum_.fillna(0), annot=annot, ax=axs[2])
    axs[0].set_title('mean')
    axs[1].set_title('count')
    axs[2].set_title('sum')
    plt.show()


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