import darkchem
from os.path import *
import numpy as np


def latent(smiles, network):
    # convert smiles
    vectors = np.vstack(darkchem.preprocess.vectorize(smiles))

    # load model
    config = darkchem.utils.load_config(join(network, 'arguments.txt'))
    config['output'] = network
    model = darkchem.utils.model_from_config(config)

    # predict latent
    latent = model.encoder.predict(vectors)

    return latent


def properties(smiles, network):
    # convert smiles
    vectors = np.vstack(darkchem.preprocess.vectorize(smiles))

    # load model
    config = darkchem.utils.load_config(join(network, 'arguments.txt'))
    config['output'] = network
    model = darkchem.utils.model_from_config(config)

    # predict latent
    latent = model.encoder.predict(vectors)

    # properties
    properties = model.predictor.predict(latent)

    return properties
