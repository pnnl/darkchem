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
    p['prep'].add_argument('-c', '--canonical', action='store_true', help='smiles column already canonicalized with openbabel')
    p['prep'].add_argument('-s', '--shuffle', action='store_true', help='shuffle input data')

    # train mode
    p['train'] = p['subparsers'].add_parser('train',
                                            description='DarkChem training module',
                                            help='configure variational autoencoder for structure/property prediction')
    p['train'].add_argument('data', type=str, help='path to input data containing vectorized smiles (str)')
    p['train'].add_argument('output', type=str, help='path to output folder (str).')
    p['train'].add_argument('--labels', '-y', type=str, default='-1', help='path to input labels (str, optional)')

    network = p['train'].add_argument_group('Network configuration')
    network.add_argument('--kernels', '-k', nargs='+', type=int, default=[10, 10, 11], help='kernel size per convolution (int, default=10 10 11)')
    network.add_argument('--filters', '-f', nargs='+', type=int, default=[9, 9, 10], help='number of filters per convolution (int, default=9 9 10)')
    network.add_argument('--embedding-dim', '-ed', type=int, default=32, help='input vector embedding dimension (int, default=32)')
    network.add_argument('--latent-dim', '-ld', type=int, default=292, help='latent encoding dimension (int, default=292)')
    network.add_argument('--epsilon', '-e', type=float, default=0.1, help='latent space standard deviation (float, default=0.1)')
    network.add_argument('--dropout', '-d', type=float, default=0.2, help='dropout fraction on property prediciton (float, default=0.2)')

    training = p['train'].add_argument_group('Training configuration')
    training.add_argument('--weights', '-w', type=str, default='-1', help='path to directory containing pretrained weights (str, optional).')
    training.add_argument('--freeze-vae', '-z', action='store_true', help='freeze autoencoder weights (optional)')
    training.add_argument('--validation', '-v', type=float, default=0.1, help='fraction to withold for validation (float, default=0.1)')
    training.add_argument('--batch-size', '-b', type=int, default=128, help='training batch size (int, default=128)')
    training.add_argument('--epochs', '-n', type=int, default=10, help='number of training epochs (int, default=10)')
    training.add_argument('--patience', '-p', type=int, default=5, help='early stopping patience (int, default=5)')
    training.add_argument('--seed', '-s', type=int, default=777, help='set random seed (int, default=777).')

    # predict mode
    p['predict'] = p['subparsers'].add_parser('predict',
                                              description='DarkChem property prediction module',
                                              help='property prediction module')
    p['predict'].add_argument('data', type=str, help='path to .tsv containing inchi/smiles for property prediction (str)')
    p['predict'].add_argument('network', type=str, help='path to saved network weights')

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
        pass

    # no module selected
    else:
        p['global'].print_help()
