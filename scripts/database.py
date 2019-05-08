import darkchem
from os.path import *
import numpy as np
import scipy.spatial
import sklearn.decomposition
import pandas as pd
import glob


def network(path):
    # load model
    config = darkchem.utils.load_config(join(path, 'arguments.txt'))
    config['output'] = path
    return darkchem.utils.model_from_config(config)


if __name__ == '__main__':

    infiles = glob.glob('../data/pubchem/pubchem_*_smiles.tsv')
    infiles.sort()

    for i, infile in enumerate(infiles):
        output = '..result/pubchem_darkchem_%03d.tsv' % i

        # # load new data
        print('Preprocessing input data for chunk %03d...' % (i + 1))
        data = pd.read_csv(infile, sep='\t', header=None, names=['SMILES'])

        can = darkchem.preprocess.inchi2smi(data['SMILES'].values)
        vec = darkchem.preprocess.vectorize(can)

        template = pd.DataFrame({'SMILES': can, 'vec': vec})

        template.dropna(how='any', inplace=True)
        vec = np.vstack(template['vec'].values)
        template = template[['InChI', 'SMILES']]

        # create data frame
        dfs = []

        # enumerate adduct types
        for adduct in ['[M+H]', '[M-H]', '[M+Na]']:
            print('Running DarkChem for %s...' % adduct)
            df = template.copy()
            df['Adduct'] = adduct

            # load train, network
            print('\tPredicting...')
            train = np.load('../data/valset_%s_smiles.npy' % adduct)
            net = network('../result/N7b_%s' % adduct)

            # predict latent
            train_latent = net.encoder.predict(train)
            test_latent = net.encoder.predict(vec)

            # predict ccs
            ccs = net.predictor.predict(test_latent)[:, -1]
            df['CCS'] = ccs

            # pca
            print('\tCalculating PCA transform...')
            p = sklearn.decomposition.PCA(n_components=8)
            train_latent_pca = p.fit_transform(train_latent)
            test_latent_pca = p.transform(test_latent)

            # convex hull
            print('\tConstructing convex hull...')
            chull = scipy.spatial.ConvexHull(train_latent_pca)
            print('\tConverting to Delaunay...')
            d = scipy.spatial.Delaunay(train_latent_pca[chull.vertices])

            # membership
            print('\tDetermining hull membership...')
            df['Hull'] = d.find_simplex(test_latent_pca) >= 0

            # add to list
            dfs.append(df)

        # save
        final = pd.concat(dfs)
        final.to_csv(output, sep='\t', index=False)
