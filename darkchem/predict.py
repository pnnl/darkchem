import darkchem
from os.path import *


def latent(smiles, network):
    # convert smiles
    vectors = darkchem.preprocess.vectorize(smiles)

    # load model
    config = darkchem.utils.load_config(join(network, 'arguments.txt'))
    config['output'] = network
    model = darkchem.utils.model_from_config(config)

    # predict latent
    latent = model.encoder.predict(vectors)

    return latent


def properties(smiles, network):
    # convert smiles
    vectors = darkchem.preprocess.vectorize(smiles)

    # load model
    config = darkchem.utils.load_config(join(network, 'arguments.txt'))
    config['output'] = network
    model = darkchem.utils.model_from_config(config)

    # predict latent
    latent = model.encoder.predict(vectors)

    # properties
    properties = model.predictor.predict(latent)

    return properties
