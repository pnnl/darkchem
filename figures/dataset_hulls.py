import darkchem
from scipy.spatial import ConvexHull
from os.path import *
import numpy as np
import pickle
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


def calculate(network, datasets, p=0.995):
    for i in range(len(datasets)):
        dataset = basename(datasets[i]).split('_')[0]

        print(dataset)

        data = np.load(datasets[i])

        # load model
        model = darkchem.utils.load_model(network)

        # latent representation
        latent = model.encoder.predict(data)

        # assumes first dataset used for pca
        if i == 0:
            pca = PCA(n_components=2, whiten=False)
            pca.fit(latent)

            # save pca object
            with open('../result/hull/pca.pkl', 'wb') as output_file:
                pickle.dump(pca, output_file)

        # downselect
        idx = darkchem.utils.downselect(latent, p=p)
        latent = latent[idx]

        x_pca = pca.transform(latent)
        hull = ConvexHull(x_pca)

        # save hull
        with open('../result/hull/%s_hull.pkl' % dataset, 'wb') as output_file:
            pickle.dump(hull, output_file)

        # save x_pca
        np.save('../result/hull/%s_x_pca.npy' % dataset, x_pca)


def plot(datasets, path):
    datasets = [basename(x).split('_')[0] for x in datasets]
    labels = ['PubChem', 'DSSTox', 'HMDB', 'UNPD', 'Experimental']

    hulls = ['../result/hull/%s_hull.pkl' % x for x in datasets]
    x_pcas = ['../result/hull/%s_x_pca.npy' % x for x in datasets]

    # load overall hull
    x_pca_all = np.load(x_pcas[0])

    with open(hulls[0], "rb") as input_file:
        hull_all = pickle.load(input_file)

    fig, ax = plt.subplots(2, 3, figsize=(7.5, 5), sharex=True, sharey=True, dpi=600)
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i in range(1, len(datasets)):
        dataset = datasets[i]
        print(dataset)
        x = (i - 1) // 3
        y = (i - 1) % 3

        # load hull
        x_pca = np.load(x_pcas[i])

        with open(hulls[i], "rb") as input_file:
            hull = pickle.load(input_file)

        # plot the big hull
        ax[x, y].fill(x_pca_all[hull_all.vertices, 0], x_pca_all[hull_all.vertices, 1],
                      alpha=0.2, label='all', zorder=1, color='grey')

        # plot the subhull
        ax[x, y].fill(x_pca[hull.vertices, 0], x_pca[hull.vertices, 1],
                      alpha=0.5, label=dataset, zorder=2, color=colors[i - 1])

        ax[x, y].set_title(labels[i - 1], fontweight='bold', size=10)

        # if (y == 0):
        #     ax[x, y].set_ylabel('PC2', fontweight='bold')
        # else:
        #     ax[x, y].set_ylabel('', fontweight='bold')

        # if (x == 1) and (y == 1):
        #     ax[x, y].set_xlabel('PC1', fontweight='bold')
        # else:
        #     ax[x, y].set_xlabel('', fontweight='bold')

        ax[x, y].tick_params(axis='both', which='both', labelbottom=False, labelleft=False,
                       bottom=False, left=False)

    fig.delaxes(ax[1, 2])

    plt.tight_layout()
    plt.savefig(path)


if __name__ == '__main__':
    network = '../result/N7b_[M+H]'
    datasets = ['../data/all_unique_smiles.npy',
                '../data/pubchem_smiles.npy',
                '../data/dsstox_[M+H]_smiles.npy',
                '../data/hmdb_[M+H]_smiles.npy',
                '../data/unpd_[M+H]_smiles.npy',
                '../data/valset_[M+H]_smiles.npy']
    # calculate(network, datasets)
    plot(datasets, '../result/figures/dataset_hulls.png')
