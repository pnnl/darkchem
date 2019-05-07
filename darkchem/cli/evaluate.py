import darkchem
import numpy as np
import pandas as pd
import argparse
from os.path import *


class Result:
    def __init__(self, path):
        # load config
        self.config = darkchem.utils.load_config(join(path, 'arguments.txt'))

        # outfile
        self.outfile = join(path, 'performance_summary.txt')

    def evaluate(self, smiles=None, labels=None, formula=False):
        # load model/data
        self.model = darkchem.utils.model_from_config(self.config)

        # set seed
        np.random.seed(self.config['seed'])

        if smiles is None:
            x, y = darkchem.utils.data_from_config(self.config)

            # test/train split
            mask = darkchem.utils.test_train_split(x, test_size=self.config['validation'])
            self.x_test = x[~mask]

            if y is not None:
                y_test = y[~mask]

        # load data from path
        else:
            self.x_test = np.load(smiles)
            if labels is not None:
                y_test = np.load(labels)
                y = y_test
            else:
                y = None

        # predict on test set
        self.latent = self.model.encoder.predict(self.x_test)
        x_pred = self.model.decoder.predict(self.latent)

        # evaluate reconstruction
        recon = x_pred.argmax(axis=-1)
        recon_error = np.mean(self.x_test != recon, axis=1)
        recon_acc = 1 - np.mean(recon_error)

        with open(self.outfile, 'w') as f:
            f.write('Reconstruction accuracy: %s\n' % recon_acc)

        # evaluate label prediction(s)
        if y is not None:
            y_pred = self.model.predictor.predict(self.latent)
            if formula is not None:
                # property error
                y_error = np.mean(np.abs(y_pred[:, :-formula] - y_test[:, :-formula]) / y_test[:, :-formula], axis=0)

                # formula error
                f_pred = np.rint(y_pred[:, -formula:])
                f_error = np.mean(np.equal(f_pred, y_test[:, -formula:]), axis=0)

                with open(self.outfile, 'a') as f:
                    f.write('Prediction error: %s\n' % y_error)
                    f.write('Formula accuracy: %s\n' % f_error)

            else:
                # property error
                y_error = np.mean(np.abs(y_pred - y_test) / y_test, axis=0)
                with open(self.outfile, 'a') as f:
                    f.write('Prediction error: %s\n' % y_error)


def main():
    # initialize parser
    parser = argparse.ArgumentParser(description='Evaluate network for (multitask) VAE.')

    # inputs/outputs
    parser.add_argument('path', type=str, help='Path to results folder (str).')
    parser.add_argument('--smiles', '-s', type=str, help='Path to SMILES to predict on.\
                                                          Must have same shape as data trained on. (str, optional).')
    parser.add_argument('--labels', '-l', type=str, help='Path to labels to predict on.\
                                                          Must have same shape as data trained on. (str, optional).')
    parser.add_argument('--formula', '-f', type=int, help='Indicate number of labels containing formula information (int, optional).')

    # parse args
    args = parser.parse_args()

    # evaluate
    r = Result(args.path)
    r.evaluate(smiles=args.smiles, labels=args.labels, formula=args.formula)


if __name__ == '__main__':
    main()
