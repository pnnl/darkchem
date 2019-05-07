import darkchem
from sklearn.decomposition import PCA
from os.path import *
import numpy as np
import pandas as pd


class Generator:
    def __init__(self, path, split=False, data=None, labels=None):
        # load model
        self.config = darkchem.utils.load_config(join(path, 'arguments.txt'))
        self.config['output'] = path
        self.model = darkchem.utils.model_from_config(self.config)

        # set seed
        np.random.seed(self.config['seed'])

        # load data
        if data is not None:
            self.config['data'] = data
        if labels is not None:
            self.config['labels'] = labels

        self.x, self.y = darkchem.utils.data_from_config(self.config)

        # latent test only
        self.latent = self.model.encoder.predict(self.x)

    def pca_analysis(self, n_components=5, data=None):
        self.pca = PCA(n_components=n_components, whiten=False)

        if data is None:
            self.x_pca = self.pca.fit(self.latent)
        else:
            self.pca.fit(self.model.encoder.predict(np.load(data)))

        self.x_pca = self.pca.transform(self.latent)

    def generate(self, mass, ccs, mass_tol=0.01, ccs_tol=0.01, n=100, k=100):
        # select mass, ccs in pc space
        idx, = np.where((mass * (1 - mass_tol) <= self.y[:, 0]) &
                        (self.y[:, 0] <= mass * (1 + mass_tol)) &
                        (ccs * (1 - ccs_tol) <= self.y[:, 1]) &
                        (self.y[:, 1] <= ccs * (1 + ccs_tol)))
        mask = np.zeros(self.x_pca.shape[0], dtype=bool)
        mask[idx] = True

        pcs = []
        for i in range(self.x_pca.shape[-1]):
            # uncorrelated
            if (self.corr['mass'][i] < 1E-3) and (self.corr['ccs'][i] < 1E-3):
                # linspace or random?
                # pc = np.linspace(mn, mx, n)
                pc = np.random.uniform(self.x_pca[:, i].min(), self.x_pca[:, i].max(), n)
                # pc = np.random.normal(loc=np.mean(self.x_pca[:, i]),
                #                       scale=np.std(self.x_pca[:, i]),
                #                       size=n)

            # correlated
            else:
                pc_sample = self.x_pca[idx, i]
                # pc = np.random.uniform(pc_sample.min(), pc_sample.max(), n)
                pc = np.random.normal(loc=np.mean(pc_sample), scale=np.std(pc_sample), size=n)
                # pc = np.random.normal(loc=np.mean(pc_sample), scale=0, size=n)

            pcs.append(pc)

        # invert pca transform
        pcs = np.array(pcs).T
        latent = self.pca.inverse_transform(pcs)

        # predict structures
        vectors = self.model.decoder.predict(latent)
        vectors = darkchem.utils.beamsearch(vectors, k=k)
        vectors = vectors.reshape(-1, vectors.shape[-1])

        # this is necessary to correctly predict mass, ccs
        re_latent = self.model.encoder.predict(vectors)
        props = self.model.predictor.predict(re_latent)

        # decode/validate structures
        structures = []
        for vec, prop in zip(vectors, props):
            smi = darkchem.utils.vec2struct(vec)
            valid = darkchem.utils.checksmi(smi)
            if valid is True:
                structures.append([smi, prop[0], prop[1]])

        df = pd.DataFrame(structures, columns=['SMILES', 'm/z', 'CCS'])
        df.drop_duplicates(subset='SMILES', inplace=True)
        return df

    def interpolate(self, smi1, smi2, space='pca', n=100, k=100, include_endpoints=True):
        input_vectors = np.vstack([darkchem.utils.struct2vec(x) for x in [smi1, smi2]])

        latent = self.model.encoder.predict(input_vectors)

        # perform interpolation in pca space
        if space.lower() == 'pca':
            pca = self.pca.transform(latent)
            v1 = pca[0, :]
            v2 = pca[1, :]
        # interpolate in latent space directly
        elif space.lower() == 'latent':
            v1 = latent[0, :]
            v2 = latent[1, :]

        # difference vector
        distance = v2 - v1
        norm = np.sqrt(np.sum(np.square(distance)))
        direction = distance / norm

        # dx
        delta = np.linspace(0, norm, num=n).reshape(-1, 1) * direction.reshape(1, -1)

        # v1 + dx
        interp = v1 + delta

        # invert pca transform
        if space.lower() == 'pca':
            interp = self.pca.inverse_transform(interp)
            v1 = self.pca.inverse_transform(v1)
            v2 = self.pca.inverse_transform(v2)

        # predict structures
        vectors = self.model.decoder.predict(interp)
        vectors = darkchem.utils.beamsearch(vectors, k=k)

        vectors = vectors.reshape(-1, vectors.shape[-1])

        if include_endpoints is True:
            vectors = np.concatenate((input_vectors[0].reshape(1, -1),
                                      vectors,
                                      input_vectors[1].reshape(1, -1)),
                                     axis=0)

        # this is necessary to correctly predict mass, ccs
        # re_latent = self.model.encoder.predict(vectors)

        # dist_v1 = np.sqrt(np.sum(np.square(re_latent - v1), axis=-1))
        # dist_v2 = np.sqrt(np.sum(np.square(re_latent - v2), axis=-1))

        # props = self.model.predictor.predict(re_latent)

        # decode/validate structures
        structures = []
        for vec in vectors:
            smi = darkchem.utils.vec2struct(vec)
            valid = darkchem.utils.checksmi(smi)
            if valid is True:
                structures.append([smi, True])
            else:
                structures.append([smi, False])

        df = pd.DataFrame(structures, columns=['SMILES', 'valid'])
        # df.drop_duplicates(subset='SMILES', inplace=True)
        return df

    def analogues(self, smiles, mode='normal', n=100, k=10, seed=0):
        np.random.seed(seed)

        input_vectors = np.vstack([darkchem.utils.struct2vec(x) for x in smiles])
        latent = self.model.encoder.predict(input_vectors)

        if mode.lower() == 'normal':
            generated = np.random.normal(loc=np.mean(latent, axis=0),
                                         scale=np.std(latent, axis=0),
                                         size=(n, latent.shape[-1]))
        elif mode.lower() == 'uniform':
            generated = np.random.uniform(low=np.min(latent, axis=0),
                                          high=np.max(latent, axis=0),
                                          size=(n, latent.shape[-1]))
        else:
            raise ValueError("keyword argument 'mode' must be 'normal' or 'uniform'")

        # # check against chull
        # chull = Delaunay(latent)
        # generated = generated[chull.find_simplex(generated) >= 0, :]

        # predict structures
        vectors = self.model.decoder.predict(generated)

        # beamsearch
        vectors = darkchem.utils.beamsearch(vectors, k=k)
        vectors = vectors.reshape(-1, vectors.shape[-1])

        # this is necessary to correctly predict mass, ccs
        re_latent = self.model.encoder.predict(vectors)
        props = self.model.predictor.predict(re_latent)

        # decode/validate structures
        structures = []
        for vec, prop in zip(vectors, props):
            smi = darkchem.utils.vec2struct(vec)
            valid = darkchem.utils.checksmi(smi)
            if valid is True:
                structures.append([smi, prop[0], prop[1]])

        # return as dataframe
        df = pd.DataFrame(structures, columns=['SMILES', 'm/z', 'CCS'])
        df.drop_duplicates(subset='SMILES', inplace=True)
        return df

    def node_traverse(self, smi1, smi2, space='latent', n=5, k=1000, d=100, epsilon=0.8):
        from scipy.spatial.distance import cdist

        input_vectors = np.vstack([darkchem.utils.struct2vec(x) for x in [smi1, smi2]])

        latent = self.model.encoder.predict(input_vectors)

        # perform interpolation in pca space
        if space.lower() == 'pca':
            pca = self.pca.transform(latent)
            v1 = pca[0, :]
            v2 = pca[1, :]
        # interpolate in latent space directly
        elif space.lower() == 'latent':
            v1 = latent[0, :]
            v2 = latent[1, :]

        # difference vector
        distance = v2 - v1
        norm = np.sqrt(np.sum(np.square(distance)))
        direction = distance / norm

        # dx
        delta = np.linspace(0, norm, num=n + 2)[1:-1].reshape(-1, 1) * direction.reshape(1, -1)

        # v1 + dx
        interp = v1 + delta

        # invert pca transform
        if space.lower() == 'pca':
            interp = self.pca.inverse_transform(interp)

        # remove query points
        nodes = np.delete(self.latent, np.argwhere((self.latent == v1) | (self.latent == v2)), axis=0)

        # find closest training point
        nodes = np.array([nodes[cdist([query], nodes).argmin()] for query in interp])

        # add starting points
        nodes = np.concatenate((latent[0].reshape(1, -1),
                                nodes,
                                latent[1].reshape(1, -1)),
                               axis=0)

        # generate noise
        noise = np.random.normal(loc=0.0, scale=epsilon, size=(k, self.latent.shape[-1]))

        # combine
        candidates = np.array([noise + x for x in nodes]).reshape((-1, self.latent.shape[-1]))

        # decode
        vectors = self.model.decoder.predict(candidates)
        vectors = darkchem.utils.beamsearch(vectors, k=d)

        vectors = vectors.reshape(-1, vectors.shape[-1])

        # this is necessary to correctly predict mass, ccs
        re_latent = self.model.encoder.predict(vectors)

        dist_v1 = np.sqrt(np.sum(np.square(re_latent - v1), axis=-1))
        dist_v2 = np.sqrt(np.sum(np.square(re_latent - v2), axis=-1))

        # decode/validate structures
        structures = []
        counter = 0
        for i, (vec, d1, d2) in enumerate(zip(vectors, dist_v1, dist_v2)):
            if (i > 0) & (i % (k * d) == 0):
                counter += 1

            smi = darkchem.utils.vec2struct(vec)
            valid = darkchem.utils.checksmi(smi)
            if valid is True:
                structures.append([smi, d1, d2, counter, True])
            else:
                structures.append([smi, d1, d2, counter, False])

        # return as dataframe
        df = pd.DataFrame(structures, columns=['SMILES', 'd1', 'd2', 'set', 'valid'])
        # df.drop_duplicates(subset='SMILES', inplace=True)
        return df


if __name__ == '__main__':
    compounds = {'adenine': 'Nc1ncnc2c1[nH]cn2',
                 'adenosine': 'OC[C@H]1O[C@H]([C@@H]([C@@H]1O)O)n1cnc2c1ncnc2N',
                 'cyclic-amp': 'O[C@@H]1[C@@H]2OP(=O)(O)OC[C@H]2O[C@H]1n1cnc2c1ncnc2N',
                 'cholesterol': 'CC(CCC[C@H]([C@H]1CC[C@@H]2[C@]1(C)CC[C@H]1[C@H]2CC=C2[C@]1(C)CC[C@@H](C2)O)C)C'}
    # compounds = pd.read_csv('~/Desktop/pcp_analogues/pcp_analogues_canonical.txt', sep='\n', header=None).values.flatten()

    g = Generator('../data/darkchem/N7b_[M+H]/',
                  data='../data/darkchem/combined_[M+H]_smiles.npy',
                  labels='../data/darkchem/combined_[M+H]_labels.npy')

    # g.pca_analysis(n_components=128)

    a = 'adenine'
    b = 'cyclic-amp'
    df = g.node_traverse(compounds[a], compounds[b],
                         n=3, k=1000, d=20, epsilon=0.1)
    df.to_csv('~/Desktop/%s-%s_3.tsv' % (a, b), sep='\t', index=False)

    # df = g.analogues(compounds, n=1000000, k=100, mode='uniform', seed=973)
    # df.to_csv('~/Desktop/pcp_analogues_1mil.tsv', sep='\t', index=False)
