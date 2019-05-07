import os
import argparse
import numpy as np


def main():
    # initialize parser
    parser = argparse.ArgumentParser(description='Configure variational autoencoder for small molecule structure.')

    # inputs/outputs
    parser.add_argument('data', type=str, help='Path to input data containing vectorized SMILES (str).')
    parser.add_argument('output', type=str, help='Path to output folder (str).')
    parser.add_argument('--labels', '-y', type=str, default='-1', help='Path to input labels (str, optional).')

    # network
    network = parser.add_argument_group('Network configuration')
    network.add_argument('--kernels', '-k', nargs='+', type=int, default=[10, 10, 11], help='Kernel size per convolution (int, default=10 10 11)')
    network.add_argument('--filters', '-f', nargs='+', type=int, default=[9, 9, 10], help='Number of filters per convolution (int, default=9 9 10)')
    network.add_argument('--embedding-dim', '-ed', type=int, default=32, help='Input vector embedding dimension (int, default=32)')
    network.add_argument('--latent-dim', '-ld', type=int, default=292, help='Latent encoding dimension (int, default=292)')
    network.add_argument('--epsilon', '-e', type=float, default=0.1, help='Latent space standard deviation (float, default=0.1)')
    network.add_argument('--dropout', '-d', type=float, default=0.2, help='Dropout fraction on property prediciton (float, default=0.2)')

    # training
    training = parser.add_argument_group('Training configuration')
    training.add_argument('--weights', '-w', type=str, default='-1', help='Path to directory containing pretrained weights (str, optional).')
    training.add_argument('--freeze-vae', '-z', action='store_true', help='Freeze autoencoder weights (optional)')
    training.add_argument('--validation', '-v', type=float, default=0.1, help='Fraction to withold for validation (float, default=0.1)')
    training.add_argument('--batch-size', '-b', type=int, default=128, help='Training batch size (int, default=128)')
    training.add_argument('--epochs', '-n', type=int, default=10, help='Number of training epochs (int, default=10)')
    training.add_argument('--patience', '-p', type=int, default=5, help='Early stopping patience (int, default=5)')
    training.add_argument('--seed', '-s', type=int, default=777, help='Set random seed (int, default=777).')

    # parse args
    args = parser.parse_args()

    # deferred import
    import darkchem

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


if __name__ == '__main__':
    main()
