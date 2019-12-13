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

    # overwrite invalids
    latent = np.where(np.all(vectors == 0, axis=1, keepdims=True), np.nan, latent)

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

    # overwrite invalids
    properties = np.where(np.all(vectors == 0, axis=1, keepdims=True), np.nan, properties)

    return properties


def softmax(smiles, network):
    # convert smiles
    vectors = np.vstack(darkchem.preprocess.vectorize(smiles))

    # load model
    config = darkchem.utils.load_config(join(network, 'arguments.txt'))
    config['output'] = network
    model = darkchem.utils.model_from_config(config)

    # predict latent
    latent = model.encoder.predict(vectors)

    # softmax
    softmax = model.decoder.predict(latent)

    # argmax and convert to smiles
    smiles_out = np.array([darkchem.utils.vec2struct(x) for x in np.argmax(softmax, axis=-1)])

    # overwrite invalids
    idx = np.where(np.all(vectors == 0, axis=1)
    smiles_out = , np.nan, smiles_out)
    softmax = np.where(np.all(vectors == 0, axis=1, keepdims=True), np.nan, softmax)

    return softmax, smiles_out
