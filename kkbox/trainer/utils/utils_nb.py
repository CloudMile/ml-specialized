import pandas as pd, seaborn as sns, re

from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt

def draw_roc_curve(y, pred):
    fprRf, tprRf, _ = roc_curve(y, pred, pos_label=1)
    auc_scr = auc(fprRf, tprRf)
    print("auc:", auc_scr)
    f, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(fprRf, tprRf, label='ROC CURVE')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Area Under Curve(ROC) (score: {:.4f})'.format(auc_scr))
    ax.legend(loc='best')
    plt.show()

def flatten(data, uni_cols:list, m_col, target):
    """Refer to `input.Input.flatten`"""
    multi = data[m_col].copy()
    multi.loc[multi.isna()] = multi[multi.isna()].map(lambda e: ('',))
    series = pd.Series(list(zip(list(data[uni_cols].values),
                                list(multi), #  .str.split(f'\s*{re.escape(sep)}\s*')),
                                list(data[target]))))

    def map_fn(e):
        uni, mul, label = e
        if type(mul) == str: raise Exception('Multi feature is string!')

        uni = tuple(uni)
        cache.extend([uni + (ele, label) for ele in mul])

    columns = uni_cols + [m_col, target]
    cache = []
    series.map(map_fn)
    return pd.DataFrame(columns=columns, data=cache)


def heatmap(data, *cols, xtick=None, ytick=None, annot=True, fmt='.2f', figsize=None):
    figsize = figsize or (16, 4)
    f, axs = plt.subplots(1, 2, figsize=figsize)

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

# def to_csv(path, df, header=True):
#     with open(path, 'w', newline='') as fp:
#         w = csv.writer(fp)
#         if header:
#             w.writerows(df.columns[None, :])
#         w.writerows(df.values)


