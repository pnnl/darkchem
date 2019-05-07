import darkchem
from scipy import stats
import numpy as np
import pickle
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_latent(data, y, points=None, bins=384, label='mass'):
    plt.figure(figsize=(2.5, 2.5), dpi=600)

    left = 0.22
    bottom = 0.2
    top = 0.9
    right = 0.9
    ax = plt.axes([left, bottom, right - left, top - bottom])
    # create axes to the top and right of the main axes and hide them
    top_ax = plt.axes([left, top, right - left, 0.98 - top])
    plt.axis('off')
    right_ax = plt.axes([right, bottom, 0.98 - right, top - bottom])
    plt.axis('off')

    if label.lower() == 'mass':
        label_idx = 0
        label_str = 'm/z'
    elif label.lower() == 'ccs':
        label_idx = 1
        label_str = 'CCS'

    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], c='red', s=1, zorder=2)

    H, xe, ye, bn = stats.binned_statistic_2d(data[:, 0],
                                              data[:, 1],
                                              y[:, label_idx],
                                              'mean', bins=bins)
    H = np.ma.masked_invalid(H)
    XX, YY = np.meshgrid(xe, ye)

    # main plot
    mesh = ax.pcolormesh(XX, YY, H.T, zorder=1, cmap='cividis')
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')

    # colorbar
    ax_in = inset_axes(ax, width='5%', height='50%', loc='upper right',
                       bbox_to_anchor=(0, 0, 0.8, 0.9),
                       bbox_transform=ax.transAxes)
    clb = plt.colorbar(mesh, cax=ax_in)
    clb.ax.set_title(label_str, fontweight='bold', loc='left', fontsize=8)

    # histograms
    sns.distplot(data[:, 0], ax=top_ax, hist=False, norm_hist=True,
                 color='#00204d', kde_kws={'shade': True})
    sns.distplot(data[:, 1], ax=right_ax, hist=False, norm_hist=True,
                 color='#00204d', kde_kws={'shade': True}, vertical=True)

    # axis limits
    ax.set_xlim(left=-15)
    # ax.set_ylim(-3, 3)
    # top_ax.set_xlim(left=-2)
    # right_ax.set_ylim(-3, 3)
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False,
                   bottom=False, left=False)

    plt.tight_layout()
    plt.savefig('../result/figures/latent_%s_pcp.png' % label)
    plt.close()


if __name__ == '__main__':
    # prep smiles
    smiles = pd.read_csv('../data/pcp_analogues_canonical.txt', sep='\n', header=None).values.flatten()
    input_vectors = np.vstack([darkchem.utils.struct2vec(x) for x in smiles])

    # load data
    data = np.load('../data/combined_[M+H]_smiles.npy')
    labels = np.load('../data/combined_[M+H]_labels.npy')

    # load model
    model = darkchem.utils.load_model('../result/N7b_[M+H]')

    # latent representation
    latent = model.encoder.predict(data)
    points = model.encoder.predict(input_vectors)
    # points = None

    # downselect
    idx = darkchem.utils.downselect(latent, p=0.995)
    latent = latent[idx]
    labels = labels[idx]

    # load pca
    with open('../result/hull/pca.pkl', 'rb') as input_file:
        pca = pickle.load(input_file)

    # transform
    x_pca = pca.transform(latent)
    points = pca.transform(points)

    # plot
    plot_latent(x_pca, labels, points=points, label='mass')
    plot_latent(x_pca, labels, points=points, label='ccs')
