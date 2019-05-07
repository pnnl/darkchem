import darkchem
import numpy as np
from os.path import *

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


def latent_dist(latent, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    tmp = [sns.distplot(x, hist=False, color='black', kde_kws={'alpha': 0.02}, ax=ax) for x in latent.T]
    ax.set_xlabel('Latent Variable', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')

    s = "$\\mu$: %.2f\n$\\sigma$: %.2f" % (latent.mean(axis=0).mean(), latent.std(axis=0).mean())
    ax.text(0.85, 0.9, s, ha='center', va='center', transform=ax.transAxes)


def latent_dists(latents, vlatents, path):
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey='row', sharex='col')
    for col, (latent, vlatent) in enumerate(zip(latents, vlatents)):
        for row, variational in enumerate([True, False]):

            if variational is True:
                latent_dist(vlatent, ax=ax[row, col])
            else:
                latent_dist(latent, ax=ax[row, col])

            if row == 1:
                ax[row, col].set_ylim(0, 2)
            if (row == 0) or (col != 1):
                ax[row, col].set_xlabel('')
            if col != 0:
                ax[row, col].set_ylabel('')

    ax[0, 0].set_title('m/z only', weight='bold')
    ax[0, 1].set_title('m/z + in silico CCS', weight='bold')
    ax[0, 2].set_title('m/z + experimental CCS', weight='bold')

    plt.tight_layout()
    plt.locator_params(axis='x', nticks=5)
    plt.locator_params(axis='y', nticks=6)
    plt.savefig(path, dpi=600)
    plt.close()


if __name__ == '__main__':
    networks = ['N1b', 'N3b_[M+H]', 'N7b_[M+H]']
    data = np.load('../data/valset_[M+H]_smiles.npy')
    # labels = np.load('data/combined_[M+H]_labels.npy')

    latents = []
    vlatents = []
    for n in networks:
        model = darkchem.utils.load_model(join('../result', n))
        latents.append(model.encoder.predict(data))
        vlatents.append(model.encoder_variational.predict(data))

    latent_dists(latents, vlatents, '../result/figures/latent_dists.png')
