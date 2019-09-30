import darkchem
import multiprocessing as mp
import re
import glob
import pandas as pd
import subprocess
import os
import numpy as np


def vectorize(smiles, processes=mp.cpu_count()):
    p = mp.Pool(processes=processes)
    return p.map(darkchem.utils.struct2vec, smiles)


def _canonicalize(smi):
    '''Canonicalizes SMILES string.'''

    try:
        res = subprocess.check_output('echo "%s" | obabel -ismi -ocan' % smi,
                                      stderr=subprocess.STDOUT, shell=True).decode('ascii')
    except:
        print(smi, 'failed.')
        return None

    res = [x.strip() for x in res.split('\n') if x is not '']

    if 'molecule converted' in res[-1]:
        return res[-2]

    return None


def canonicalize(smiles, processes=mp.cpu_count()):
    p = mp.Pool(processes=processes)
    return p.map(_canonicalize, smiles)


def _inchi2smi(inchi):
    '''Converts InChI string to SMILES string.'''

    try:
        res = subprocess.check_output('echo "%s" | obabel -iinchi -ocan' % inchi,
                                      stderr=subprocess.STDOUT, shell=True).decode('ascii')
    except:
        print(inchi, 'failed.')
        return None

    res = [x.strip() for x in res.split('\n') if x is not '']

    if 'molecule converted' in res[-1]:
        return res[-2]

    return None


def inchi2smi(inchis, processes=mp.cpu_count()):
    p = mp.Pool(processes=processes)
    return p.map(_inchi2smi, inchis)


def _parse_formula(formula, targets='CHNOPS'):
    atoms = re.findall(r'([A-Z][a-z]?)(\d+)?', formula)

    d = {k: v for k, v in atoms if k in targets}

    for k in targets:
        if k in d.keys():
            if d[k] == '':
                d[k] = 1
            else:
                d[k] = int(d[k])
        else:
            d[k] = 0
    return d


def parse_formulas(formulas, processes=mp.cpu_count()):
    p = mp.Pool(processes=processes)
    return pd.DataFrame(data=p.map(_parse_formula, formulas))


def process(df, name, output, canonical=False, shuffle=True):
    '''
    Assumes dataframe with InChI or SMILES columns and
    optionally a Formula column.  Any additional columns will
    be propagated as labels for prediction.
    '''

    # shuffle data
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # already converted
    if 'SMILES' in df.columns and canonical is True:
        pass
    # convert inchi to canonical smiles
    elif 'InChI' in df.columns and 'SMILES' not in df.columns:
        df['SMILES'] = inchi2smi(df['InChI'].values)
        df.to_csv(os.path.join(output, '%s_smiles.tsv' % name), index=False, sep='\t')
    # canonicalize existing smiles
    elif 'SMILES' in df.columns:
        df['SMILES'] = canonicalize(df['SMILES'].values)
        df.to_csv(os.path.join(output, '%s_canonical.tsv' % name), index=False, sep='\t')
    # error
    else:
        raise KeyError('Dataframe must have an "InChI" or "SMILES" column.')

    # vectorize
    # TODO: configurable max length
    # TODO: configurable charsest
    vectors = np.vstack(darkchem.preprocess.vectorize(df['SMILES'].values))
    vectors = np.where(np.all(vectors == 0, axis=1, keepdims=True), np.nan, vectors)

    df['vec'] = vectors.tolist()

    df.dropna(how='any', axis=0, inplace=True)
    arr = np.vstack(df['vec'].values)

    # labels
    if 'InChI' in df.columns:
        labels = df.drop(columns=['InChI', 'SMILES', 'vec'])
    else:
        labels = df.drop(columns=['SMILES', 'vec'])

    # save
    np.save(os.path.join(output, '%s.npy' % name), arr)

    if len(labels.columns) > 0:
        np.save(os.path.join(output, '%s_labels.npy' % name), labels.values)


class SDF:
    def __init__(self):
        pass

    def parse(folder, dest):
        files = glob.glob(os.path.join(folder, '*.sdf'))

        files.sort()

        n = len(files)
        with open(dest, 'w') as outfile:
            outfile.write('Formula\tInChI\tSMILES\tMonoisotopic Weight\n')

            for i, f in enumerate(files):
                print(os.path.basename(f), '(%s - %.2f%%)' % (i, 100 * i / n))

                with open(f) as text:

                    inchi = None
                    smiles = None
                    weight = None
                    formula = None

                    for line in text:
                        if '<PUBCHEM_IUPAC_INCHI>' in line:
                            nl = next(text).strip()
                            inchi = nl
                        elif '<PUBCHEM_MOLECULAR_FORMULA>' in line:
                            nl = next(text).strip()
                            formula = nl
                        elif '<PUBCHEM_MONOISOTOPIC_WEIGHT>' in line:
                            nl = next(text).strip()
                            weight = float(nl)
                        elif '<PUBCHEM_OPENEYE_CAN_SMILES>' in line:
                            nl = next(text).strip()
                            smiles = nl
                        elif '$$$$' in line:
                            if None not in [formula, weight, inchi, smiles]:
                                outfile.write('%s\t%s\t%s\t%s\n' % (formula, inchi, smiles, weight))

                            inchi = None
                            smiles = None
                            weight = None
                            formula = None

    def filter(infile, dest):
        re_formula = re.compile(r'^[CHNOPS0-9]+$').search
        re_inchi = re.compile(r'[pq\.]').search
        re_c = re.compile(r'[C]').search

        def check_formula(s):
            return (bool(re_formula(s)) and bool(re_c(s)))

        def check_inchi(s):
            return not bool(re_inchi(s))

        with open(dest, 'w') as outfile:
            outfile.write('Formula\tInChI\tSMILES\tMonoisotopic Weight\n')

            with open(infile, 'r') as f:
                for i, line in enumerate(f):
                    if i > 0:
                        formula, inchi, smiles, mass = line.strip().split('\t')
                        if check_formula(formula) and check_inchi(inchi) and float(mass) < 1000:
                            outfile.write('%s\t%s\t%s\t%s\n' % (formula, inchi, smiles, mass))
                            print(formula, 'pass')
                        else:
                            print(formula, 'fail')

    def filter2(infile, dest):
        re_formula = re.compile(r'^[CHNOPS0-9]+$').search
        re_inchi = re.compile(r'[pq\.]').search
        re_c = re.compile(r'[C]').search

        def check_formula(s):
            return (bool(re_formula(s)) and bool(re_c(s)))

        def check_inchi(s):
            return not bool(re_inchi(s))

        with open(dest, 'w') as outfile:

            with open(infile, 'r') as f:
                for i, line in enumerate(f):
                    if i > 0:
                        formula, inchi, smiles, mass = line.strip().split('\t')
                        if check_formula(formula) and check_inchi(inchi) and float(mass) < 1000:
                            outfile.write('%s\n' % smiles)
                            print(formula, 'pass')
                        else:
                            print(formula, 'fail')
