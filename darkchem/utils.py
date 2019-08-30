import numpy as np
import os
import ast
from rdkit import Chem
import multiprocessing as mp
from functools import partial
from os.path import *
import keras

# Globals
SMI = ['PAD',
       '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
       'c', 'h', 'C', 'H', 'N', 'O', 'P', 'S',
       '/', '-', '(', ')', ',',
       'n', 'o', 'p', 's', '.', '=', '#', '$', ':', '\\', '@', '+', '[', ']'  # add'l smi chars
       ]


def load_model(dirname):
    return model_from_config(load_config(join(dirname, 'arguments.txt')))


def load_config(filepath):
    config = {}
    with open(filepath) as f:
        for line in f:
            (key, val) = [x.strip() for x in line.split(':')]
            if key == '-1':
                config[key] = None
            elif key in ['nchars', 'max_length', 'embedding_dim', 'nlabels',
                         'latent_dim', 'batch_size', 'epochs', 'patience', 'seed']:
                config[key] = int(val)
            elif key in ['kernels', 'filters', 'freeze_vae']:
                config[key] = ast.literal_eval(val)
            elif key in ['epsilon', 'dropout', 'validation']:
                config[key] = float(val)
            elif key in ['data', 'output', 'weights', 'labels']:
                config[key] = val

    config['epsilon_std'] = config.pop('epsilon')
    config['output'] = dirname(filepath)
    return config


def model_from_config(config):
    # initialize autoencoder
    from darkchem.network import VAE
    model = VAE()
    if config['labels'] is not None:
        model.create_multitask(**config)
    else:
        model.create(**config)

    # load weights
    model.encoder.load_weights(os.path.join(config['output'], 'encoder.h5'))
    model.decoder.load_weights(os.path.join(config['output'], 'decoder.h5'))
    if config['labels'] is not None:
        model.predictor.load_weights(os.path.join(config['output'], 'predictor.h5'))

    return model


def data_from_config(config):
    x = np.load(config['data'])

    if config['labels'] is not None:
        y = np.load(config['labels'])
        return x, y
    else:
        return x, None


def test_train_split(x, test_size=0.1):

    idx = np.random.choice(np.arange(len(x)), size=int(len(x) * test_size), replace=False)
    mask = np.ones(len(x)).astype('bool')
    mask[idx] = False

    return mask


def _encode(string, charset):
    '''
    Encodes string with a given charset.
    Returns None if s contains illegal characters
    If s is empty, returns an empty array
    '''

    if string is None or string is np.nan:
        return np.array([])

    vec = np.zeros(len(string))
    for i in range(len(string)):
        s = string[i]
        if s in charset:
            vec[i] = charset.index(s)
        else:  # Illegal character in s
            return None

    return vec


def _add_padding(l, length):
    '''
    Adds padding to l to make it size length.
    '''

    ltemp = list(l)
    ltemp.extend([0] * (length - len(ltemp)))
    return ltemp


def _smi2vec(smi, charset, max_length):
    # Encode SMILES
    vec = _encode(smi, charset)

    # Check for errors
    if vec is None:
        # print('%s skipped, contains illegal characters' % smi)
        return None
    if len(vec) > max_length:
        # print('%s skipped, too long' % smi)
        return None

    # Add padding
    vec = _add_padding(vec, max_length)

    # Return encoded InChI
    return vec


def struct2vec(struct, charset=SMI, max_length=100):
    '''
    Takes in structure and returns the encoded version using the default
    or passed in charset.

    Parameters
    ----------
    struct : str
        Structure of compound, represented as an InChI or SMILES string.

    charset : list, optional
        Character set used for encoding.

    max_length : int, optional
        Maximum length of encoding.

    Returns
    -------
    vec : unit8 array
        Encoded structure
    '''

    output = _smi2vec(struct, charset, max_length)

    if output is None:
        return np.zeros(max_length)
    else:
        return np.array(output, dtype=np.uint8)


def vec2struct(vec, charset=SMI):
    '''
    Decodes a structure using the given charset.

    Parameters
    -------
    vec : unit8 array
        Encoded structure

    charset : list, optional
        Character set used for encoding.

    Returns
    -------
    struct : str
        Structure of compound, represented as an InChI or SMILES string.
        Note: for InChIs, no layer past the hydrogen layer will be
        available. All conformers are possible.
    '''

    # Init
    struct = ''
    for i in vec:
        try:
            # Place character
            if charset[i] != 'PAD':
                struct += charset[i]
        except:
            raise KeyError('Invalid character encountered.')
            return None

    # Return decoded structure.
    return struct


def savedict(d, path, verbose=True):
    if verbose:
        print('Arguments:')
    with open(os.path.join(path, 'arguments.txt'), 'w') as f:
        for k, v in d.items():
            if v is None:
                f.write("%s: %s\n" % (k, '-1'))
            else:
                f.write("%s: %s\n" % (k, v))
            if verbose:
                print("\t%s: %s" % (k, v))


class DataGenerator(object):
    'Generates data for Keras'

    def __init__(self, charset=SMI, batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.charset = charset

    def generate(self, partitions, labels=None):
        'Generates batches of samples'
        import keras

        if labels is None:
            labels = [None for partition in partitions]

        # Infinite loop
        while 1:
            for partition, label in zip(partitions, labels):
                # load partition
                X = np.load(partition)
                n, m = X.shape

                # generate labels
                y1 = keras.utils.to_categorical(X, len(self.charset))
                y1 = y1.reshape((-1, m, len(self.charset)))

                if label is not None:
                    y2 = np.load(label)

                # indices
                idx = np.arange(n)
                if self.shuffle:
                    np.random.shuffle(idx)

                # iterate batches
                batches = n // self.batch_size
                for i in range(batches):
                    X_batch = X[idx[i * self.batch_size:(i + 1) * self.batch_size], :]
                    y_batch = y1[idx[i * self.batch_size:(i + 1) * self.batch_size], :, :]

                    if label is not None:
                        y2_batch = y2[idx[i * self.batch_size:(i + 1) * self.batch_size], :]
                        y_batch = [y_batch, y2_batch]

                    yield X_batch, y_batch


def checksmi(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    else:
        return True


def beamsearch(data, k=10, processes=mp.cpu_count()):
    with mp.Pool(processes=processes) as pool:
        return np.array(pool.map(partial(_beamsearch, k=k), data))


def _beamsearch(instance, k=10):
    # initialize sequence container
    sequences = [[list(), 0.0]]

    # walk over each step in sequence
    for charslot in instance:
        candidates = list()

        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]

            # iterate over possible chars
            for j in range(len(charslot)):
                # log prevents probability underflow
                candidate = [seq + [j], score + np.log(charslot[j])]
                candidates.append(candidate)

        # order all candidates by score
        ordered = sorted(candidates, key=lambda tup: tup[1])

        # select k best
        sequences = ordered[-k:]

    # separate sequences from scores
    return np.array([x[0] for x in sequences])


def downselect(data, p=0.95):
    # find center
    center = np.mean(data, axis=0)

    # radial distance
    rdist = np.sqrt(np.sum(np.square(data - center), axis=-1))

    # sort
    idx = np.argsort(rdist)

    # filter by percentile
    return idx[:int(p * idx.shape[0])]


def evaluate(data, network, labels=None, validation=None, seed=777):
    result = {}
    model = load_model(network)

    x = np.load(data)

    # validation subset
    if validation is not None:
        # set seed
        np.random.seed(seed)

        # split
        mask = test_train_split(x, validation)

        # grab validation
        x = x[~mask]

    # one hot
    n, m = x.shape
    d = max(np.unique(x)) + 1
    one_hot = keras.utils.to_categorical(x, d).reshape((-1, m, d))

    # predict
    latent = model.encoder.predict(x)
    decoded = model.decoder.predict(latent)

    # reconstruction accuracy
    result['reconstruction accuracy'] = [100 * np.mean(np.equal(np.argmax(one_hot, axis=-1),
                                                                np.argmax(decoded, axis=-1)))]

    # property prediction
    if labels is not None:
        y = np.load(labels)

        # validation subset
        if validation is not None:
            y = y[~mask]

        # predict
        y_hat = model.predictor.predict(latent)

        # property prediction error
        result['property predicton error'] = 100 * np.mean(np.abs(y - y_hat) / y, axis=0)

    # display results
    for k, v in result.items():
        v = '\t'.join(['%.3f%%' % x for x in v])
        print('%s:\t%s' % (k, v))
