import numpy as np
import pandas as pd
import glob
import os

if __name__ == '__main__':
    fs = glob.glob('data/pubchem/*_smiles.npy')
    ls = glob.glob('data/pubchem/*_labels.npy')
    fs.sort()
    ls.sort()

    res = []
    for f, l in zip(fs, ls):
        res.append([os.path.basename(f), os.path.basename(l), len(np.load(f))])

    df = pd.DataFrame(res)
    df.to_csv('data/pubchem/index.tsv', sep='\t', header=None, index=None)
