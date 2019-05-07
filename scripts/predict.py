import darkchem
from os.path import *
import numpy as np
import pandas as pd


def predict(network, molecules):
    # load model
    config = darkchem.utils.load_config(join(network, 'arguments.txt'))
    config['output'] = network
    model = darkchem.utils.model_from_config(config)

    return model.predictor.predict(model.encoder.predict(molecules))


if __name__ == '__main__':
    molecules = np.load('/Users/colb804/Desktop/sumner_molecules_smiles.npy')
    print(molecules.shape)
    smi = [darkchem.utils.vec2struct(x) for x in molecules]

    df = pd.DataFrame({'SMILES': smi})

    for adduct in ['[M+H]', '[M-H]', '[M+Na]']:
        res = predict('/Users/colb804/workspace/data/darkchem/N7b_%s' % adduct, molecules)
        df[adduct] = res[:, -1]

    inchi = pd.read_csv('/Users/colb804/Desktop/sumner_molecules_smiles.tsv', sep='\t')

    df = pd.merge(df, inchi, on='SMILES')
    df = df[['InChI', 'SMILES', '[M+H]', '[M-H]', '[M+Na]']]

    df.to_csv('/Users/colb804/Desktop/sumner_darkchem.tsv', sep='\t', index=False)

    print(len(df.index))
