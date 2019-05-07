import darkchem
import numpy as np
from scipy import stats
import pandas as pd
import pickle

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


def corrplot(data, y, path, space='latent', func=np.argmax):
    idx1 = func([stats.linregress(data[:, i], y[:, 0])[2]**2 for i in range(data.shape[-1])])
    idx2 = func([stats.linregress(data[:, i], y[:, 1])[2]**2 for i in range(data.shape[-1])])
    df = pd.DataFrame({'x1': data[:, idx1], 'x2': data[:, idx2], 'CCS': y[:, 1], 'm/z': y[:, 0]})

    _, _, r1, _, _ = stats.linregress(df['x1'], df['m/z'])
    _, _, r2, _, _ = stats.linregress(df['x2'], df['CCS'])

    fig, ax = plt.subplots(1, 2, figsize=(5, 2.5), dpi=600, sharex=True)

    ax[0].set_ylabel('m/z', fontweight='bold', fontstyle='italic')
    ax[0].set_ylim((0, 1000))
    ax[1].set_ylabel('CCS', fontweight='bold')
    ax[1].set_ylim((0, 400))

    if space.lower() == 'latent':
        if func == np.argmax:
            ax[0].set_xlim(-4.5, 2)
            ax[1].set_xlim(-4.5, 2)
        elif func == np.argmin:
            ax[0].set_xlim(-2.25, 2.25)
            ax[1].set_xlim(-2.25, 2.25)
    elif space.lower() == 'pca':
        ax[0].set_xlim(-20, 25)
        ax[1].set_xlim(-20, 25)

    plt.locator_params(nbins=4, axis='x')

    sns.regplot(x='x1', y='m/z', data=df, ax=ax[0], ci=None,
                scatter_kws={'zorder': 1, 'color': 'gray', 'edgecolors': 'white', 'linewidths': 0.2, 's': 15},
                line_kws={'zorder': 2, 'color': 'black', 'linestyle': '--', 'label': '$r^2$: %.2f' % r1**2})

    sns.regplot(x='x2', y='CCS', data=df, ax=ax[1], ci=None,
                scatter_kws={'zorder': 1, 'color': 'gray', 'edgecolors': 'white', 'linewidths': 0.2, 's': 15},
                line_kws={'zorder': 2, 'color': 'black', 'linestyle': '--', 'label': '$r^2$: %.2f' % r2**2})

    if space.lower() == 'pca':
        ax[0].set_xlabel('PC1', fontweight='bold')
        ax[1].set_xlabel('PC1', fontweight='bold')
        ax[0].legend(loc='upper left', frameon=False, fontsize='x-small')
        ax[1].legend(loc='upper left', frameon=False, fontsize='x-small')
    elif space.lower() == 'latent':
        ax[0].set_xlabel('LD%s' % idx1, fontweight='bold')
        ax[1].set_xlabel('LD%s' % idx2, fontweight='bold')
        ax[0].legend(loc='upper right', frameon=False, fontsize='x-small')
        ax[1].legend(loc='upper right', frameon=False, fontsize='x-small')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    data = np.load('../data/valset_[M+H]_SMILES.npy')
    labels = np.load('../data/valset_[M+H]_labels.npy')

    # load model
    model = darkchem.utils.load_model('../result/N7b_[M+H]')

    # latent representation
    latent = model.encoder.predict(data)

    # load pca
    with open('../result/hull/pca.pkl', "rb") as input_file:
        pca = pickle.load(input_file)

    # pca transform
    x_pca = pca.transform(latent)

    corrplot(latent, labels, '../result/figures/correlation_latent_max.png', space='latent', func=np.argmax)
    corrplot(latent, labels, '../result/figures/correlation_latent_min.png', space='latent', func=np.argmin)
    corrplot(x_pca, labels, '../result/figures/correlation_pca_max.png', space='pca', func=np.argmax)
