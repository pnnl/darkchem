import darkchem
import numpy as np
import pickle
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute(network, data, path):
    # load data
    data = np.load(data)

    # load model
    model = darkchem.utils.load_model(network)

    # latent representation
    latent = model.encoder.predict(data)

    pca = PCA(n_components=128, whiten=False)
    pca.fit(latent)

    # save pca object
    with open(path, 'wb') as output_file:
        pickle.dump(pca, output_file)


def scree(pca, path):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
    ax.plot(np.arange(pca.n_components_), pca.explained_variance_ratio_, label='Component', c='#2d91a7')
    ax.plot(np.arange(pca.n_components_), np.cumsum(pca.explained_variance_ratio_), label='Cumulative', c='#dc290c')

    idx = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0]

    ax.axvline(np.arange(pca.n_components_)[idx], linestyle='--', color='k', label='90% Explained')

    plt.xlabel('Principal Component Number', fontweight='bold')
    plt.ylabel('Explained Variance', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    # compute('../result/N7b_[M+H]', '../data/combined_[M+H]_smiles.npy', '../result/pca/pca.pkl')

    # load pca
    with open('../result/pca/pca.pkl', "rb") as input_file:
        pca = pickle.load(input_file)

    scree(pca, '../result/figures/scree.png')
