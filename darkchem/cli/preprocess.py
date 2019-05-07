import darkchem.preprocess
import pandas as pd
import numpy as np
import os
import argparse


def process(df, name, output, canonical=False, shuffle=True):
    '''
    Assumes dataframe with InChI or SMILES columns and
    optionally a Formula column.  Any additional columns will
    be propagated as labels for prediction.
    '''

    # shuffle data
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # already converted
    if 'SMILES' in df.columns and canonical is True:
        pass
    # convert inchi to canonical smiles
    elif 'InChI' in df.columns and 'SMILES' not in df.columns:
        df['SMILES'] = darkchem.preprocess.inchi2smi(df['InChI'].values)
        df.to_csv(os.path.join(output, '%s_smiles.tsv' % name), index=False, sep='\t')
    # canonicalize existing smiles
    elif 'SMILES' in df.columns:
        df['SMILES'] = darkchem.preprocess.canonicalize(df['InChI'].values)
        df.to_csv(os.path.join(output, '%s_canonical.tsv' % name), index=False, sep='\t')
    # error
    else:
        raise ValueError('Dataframe must have an "InChI" or "SMILES" column.')

    # vectorize
    # TODO: configurable max length
    # TODO: configurable charsest
    df['vec'] = darkchem.preprocess.vectorize(df['SMILES'].values)
    df.dropna(how='any', axis=0, inplace=True)
    arr = np.vstack(df['vec'].values)

    # labels
    if 'InChI' in df.columns:
        labels = df.drop(columns=['InChI', 'SMILES', 'vec'])
    else:
        labels = df.drop(columns=['SMILES', 'vec'])

    # save
    np.save(os.path.join(output, '%s_smiles.npy' % name), arr)

    if len(labels.columns) > 0:
        np.save(os.path.join(output, '%s_labels.npy' % name), labels.values)


def main():
    # initialize parser
    parser = argparse.ArgumentParser(description='Preprocess data for (multitask) VAE.')

    # inputs/outputs
    parser.add_argument('data', type=str, help='Path to input .tsv file containing InChI or SMILES column (str).')
    parser.add_argument('output', type=str, help='Path to output folder (str).')
    parser.add_argument('-c', '--canonical', action='store_true', help='SMILES column already canonicalized with OpenBabel.')
    parser.add_argument('-s', '--shuffle', action='store_true', help='Shuffle input data.')

    # parse args
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.data))[0]

    df = pd.read_csv(args.data, sep='\t')

    process(df, name, args.output, canonical=args.canonical, shuffle=args.shuffle)


if __name__ == '__main__':
    main()
