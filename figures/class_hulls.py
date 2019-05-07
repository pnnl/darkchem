import darkchem
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
import pickle

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute():
    # load data
    data = np.load('../data/classes/classes_smiles.npy')
    labels = np.load('../data/classes/classes_labels.npy')

    # load model
    model = darkchem.utils.load_model('../result/N7b_[M+H]')

    # latent representation
    latent = model.encoder.predict(data)

    # # downselect
    # idx = darkchem.utils.downselect(latent, p=0.995)
    # latent = latent[idx]
    # labels = labels[idx]

    # load pca
    with open('../result/hull/pca.pkl', 'rb') as input_file:
        pca = pickle.load(input_file)

    # transform
    x_pca = pca.transform(latent)

    # save
    np.save('../data/classes/classes_x_pca.npy', x_pca)
    np.save('../data/classes/classes_x_pca_labels.npy', labels)


def class_plot(x_pca, y, hull_vertices, path, level='superclass'):

    delaunay = Delaunay(hull_vertices)

    def convex_hull(x=None, y=None, hue=None, data=None, color=None, label=None):
        data = np.stack((x, y), axis=1)

        # in the hull
        data = data[delaunay.find_simplex(data) >= 0]

        idx = darkchem.utils.downselect(data, p=0.995)
        data = data[idx]

        hull = ConvexHull(data)
        plt.fill(data[hull.vertices, 0], data[hull.vertices, 1], alpha=0.8, color=color, label=label)

    if level.lower() == 'superclass':
        level = 'Super Class'
    elif level.lower() == 'class':
        level = 'Class'

    df = pd.DataFrame({'PC1': x_pca[:, 0], 'PC2': x_pca[:, 1],
                       'Class': y[:, 1], 'Super Class': y[:, 0]})

    # combine small groups
    count = df.groupby(level).size().reset_index(name='count')
    df = pd.merge(df, count, on='Super Class')
    df.loc[df['count'] < 100, 'Super Class'] = 'zzzOther'

    df.loc[df['Super Class'] == 'Homogeneous non-metal compounds', 'Super Class'] = 'zzzOther'
    df.loc[df['Super Class'] == 'Hydrocarbon derivatives', 'Super Class'] = 'Hydrocarbons and derivatives'
    df.loc[df['Super Class'] == 'Hydrocarbons', 'Super Class'] = 'Hydrocarbons and derivatives'

    df.loc[df['Super Class'] == 'Organooxygen compounds', 'Super Class'] = 'Organooxygen and organic oxygen compounds'
    df.loc[df['Super Class'] == 'Organic oxygen compounds', 'Super Class'] = 'Organooxygen and organic oxygen compounds'

    df.sort_values(by=['Super Class', 'Class'], inplace=True)
    df['Super Class'].replace({'compounds': ''}, inplace=True, regex=True)

    classes = ['Alkaloids', 'Benzenoids', 'Hydrocarbons',
               '(Neo)lignans', 'Lipids',
               'Nucleosides,\nnucleotides', 'Organic\n1,3-dipolar',
               'Organic acids', 'Organic nitrogen',
               'Organo-\nheterocyclic', 'Organometallic',
               'Organooxygen', 'Organophosphorus',
               'Organosulfur', 'Phenylpropanoids,\npolyketides', 'Other']

    df.loc[df['Super Class'] == 'zzzOther', 'Super Class'] = 'Other'

    plt.figure(dpi=600)
    g = sns.FacetGrid(df, col=level, hue=level, col_wrap=4, height=1, aspect=1,
                      legend_out=False, sharex=True, sharey=True)
    g = g.map(convex_hull, "PC1", "PC2")

    g.set_titles(row_template='', col_template='')
    g.set_axis_labels('', '')

    for i, ax in enumerate(g.axes):
        ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], alpha=0.2, color='gray')
        ax.set_title(classes[i], wrap=True, fontsize=6, weight='bold')
        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False,
                       bottom=False, left=False)

    # plt.tight_layout()
    plt.savefig(path, dpi=600, bbox_inches="tight")


if __name__ == '__main__':
    # compute()

    # load
    x_pca = np.load('../data/classes/classes_x_pca.npy')
    labels = np.load('../data/classes/classes_x_pca_labels.npy')

    # load hull
    with open('../result/hull/all_hull.pkl', "rb") as input_file:
        hull = pickle.load(input_file)

    points = np.load('../result/hull/all_x_pca.npy')
    vertices = points[hull.vertices]

    class_plot(x_pca, labels, vertices, '../result/figures/classes.png')
