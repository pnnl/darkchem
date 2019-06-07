import argparse
import darkchem
import os
import numpy as np
import pandas as pd


def main():
    p = {}

    # overall
    p['global'] = argparse.ArgumentParser(description='DarkChem: deep learning to elucidate chemical dark matter')
    p['global'].add_argument('-v', '--version', action='version', version=darkchem.__version__, help='print version and exit')

    # subparsers
    p['subparsers'] = p['global'].add_subparsers(title='commands', dest='which')

    # prep mode
    p['prep'] = p['subparsers'].add_parser('prep',
                                           description='DarkChem input preparation module',
                                           help='preprocess data for (multitask) vae')
    p['prep'].add_argument('data', type=str, help='path to input .tsv file containing inchi or smiles column (str)')
    p['prep'].add_argument('output', type=str, help='path to output folder (str)')
    p['prep'].add_argument('-c', '--canonical', action='store_true', help='indicate smiles column already canonicalized with openbabel')
    p['prep'].add_argument('-s', '--shuffle', action='store_true', help='shuffle input data')

    # train mode
    p['train'] = p['subparsers'].add_parser('train',
                                            description='DarkChem training module',
                                            help='configure (multitask) vae for structure/property prediction')
    p['train'].add_argument('data', type=str, help='path to input data containing vectorized smiles (str)')
    p['train'].add_argument('output', type=str, help='path to output folder (str).')
    p['train'].add_argument('-y', '--labels', type=str, default='-1', help='path to input labels (str, optional)')

    network = p['train'].add_argument_group('Network configuration')
    network.add_argument('-k', '--kernels', nargs='+', type=int, default=[9, 9, 10], help='kernel size per convolution (int, default=9 9 10)')
    network.add_argument('-f', '--filters', nargs='+', type=int, default=[10, 10, 11], help='number of filters per convolution (int, default=10 10 11)')
    network.add_argument('-ed', '--embedding-dim', type=int, default=32, help='input vector embedding dimension (int, default=32)')
    network.add_argument('-ld', '--latent-dim', type=int, default=292, help='latent encoding dimension (int, default=292)')
    network.add_argument('-e', '--epsilon', type=float, default=0.1, help='latent space standard deviation (float, default=0.1)')
    network.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout fraction on property prediciton (float, default=0.2)')

    training = p['train'].add_argument_group('Training configuration')
    training.add_argument('-w', '--weights', type=str, default='-1', help='path to directory containing pretrained weights (str, optional).')
    training.add_argument('-z', '--freeze-vae', action='store_true', help='freeze autoencoder weights (optional)')
    training.add_argument('-v', '--validation', type=float, default=0.1, help='fraction to withold for validation (float, default=0.1)')
    training.add_argument('-b', '--batch-size', type=int, default=128, help='training batch size (int, default=128)')
    training.add_argument('-n', '--epochs', type=int, default=10, help='number of training epochs (int, default=10)')
    training.add_argument('-p', '--patience', type=int, default=5, help='early stopping patience (int, default=5)')
    training.add_argument('-s', '--seed', type=int, default=777, help='set random seed (int, default=777).')

    # predict mode
    p['predict'] = p['subparsers'].add_parser('predict',
                                              description='DarkChem property prediction module',
                                              help='prediction module')
    p['predict'].add_argument('mode', choices=['latent', 'prop'], help='prediction mode (str)')
    p['predict'].add_argument('data', type=str, help='path to .tsv containing inchi/smiles for property prediction (str)')
    p['predict'].add_argument('network', type=str, help='path to saved network weights (str)')
    p['predict'].add_argument('-c', '--canonical', action='store_true', help='indicate smiles column already canonicalized with openbabel')

    args = p['global'].parse_args()

    # input processing
    if args.which == 'prep':
        name = os.path.splitext(os.path.basename(args.data))[0]
        df = pd.read_csv(args.data, sep='\t')

        darkchem.preprocess.process(df, name, args.output, canonical=args.canonical, shuffle=args.shuffle)

    # train module
    elif args.which == 'train':
        # save args
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        # set seed
        np.random.seed(args.seed)

        # process single file input
        if os.path.isfile(args.data):
            darkchem.train(args)

        # process folder input (generator)
        else:
            darkchem.train_generator(args)

    # predict module
    elif args.which == 'predict':
        df = pd.read_csv(args.data, sep='\t')

        name = os.path.splitext(os.path.basename(args.data))[0]

        # already converted
        if 'SMILES' in df.columns and args.canonical is True:
            smiles = df['SMILES'].values
        # convert inchi to canonical smiles
        elif 'InChI' in df.columns and 'SMILES' not in df.columns:
            print('converting inchi to smiles...')
            df['SMILES'] = darkchem.utils.inchi2smi(df['InChI'].values)
            smiles = df['SMILES'].values
        # canonicalize existing smiles
        elif 'SMILES' in df.columns:
            print('canonicalizing smiles...')
            df['SMILES_canonical'] = darkchem.preprocess.canonicalize(df['SMILES'].values)
            smiles = df['SMILES_canonical'].values
        # error
        else:
            raise ValueError('data frame must have an "InChI" or "SMILES" column')

        # property prediction
        if args.mode == 'prop':
            print('predicting properties...')
            properties = darkchem.predict.properties(smiles, args.network)
            for i in range(len(properties.shape[-1])):
                df['prop%03d' % i] = properties[:, i]
            df.to_csv('%s_darkchem.tsv' % name, sep='\t', index=False)

        # latent prediction
        elif args.mode == 'latent':
            print('predicting latent representation...')
            latent = darkchem.predict.latent(smiles, args.network)
            np.save('%s_latent.npy' % name, latent)

    # no module selected
    else:
        p['global'].print_help()
