import glob
from os.path import *
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot(df, key='val_loss', ax=None, logx=False, logy=False):
    if ax is None:
        fig, ax = plt.subplots()

    x = np.log10(df.index + 1)
    y1 = df[key]
    y2 = df[key.split('_', 1)[-1]]

    # remap
    f1 = interp1d(x, y1)
    f2 = interp1d(x, y2)
    x_prime = np.linspace(x.min(), x.max(), len(x))
    y1_prime = f1(x_prime)
    y2_prime = f2(x_prime)

    # validation
    l1 = lowess(y1_prime, x_prime, frac=0.1, it=3, is_sorted=True, return_sorted=False)
    ax.plot(10**x_prime, l1, color='#dc290c', label='Validation')
    ax.plot(10**x, y1, color='#dc290c', alpha=0.2, label='')

    # training
    l2 = lowess(y2_prime, x_prime, frac=0.1, it=3, is_sorted=True, return_sorted=False)
    ax.plot(10**x_prime, l2, color='#2d91a7', label='Train')
    ax.plot(10**x, y2, color='#2d91a7', alpha=0.2, label='')

    if logx is True:
        ax.set_xscale('log')
    if logy is True:
        ax.set_yscale('log')

    ax.set_xlabel('epoch', fontweight='bold')
    ax.set_ylabel(key.split('_', 1)[-1].replace('_', ' '), fontweight='bold')


def load_log(path):
    log = [join(path, x) for x in glob.glob1(path, '*.out')]

    if len(log) > 1:
        print('Warning: more than one log found.')

    with open(log[0], 'r') as f:
        lines = [x.strip().split(' - ') for x in f.readlines()]

    parsed = [[x.strip().split(':') for x in row if x is not ''][1:] for row in lines]
    parsed = [x for x in parsed if len(x) > 0]
    res = []
    for p in parsed:
        try:
            res.append({k: float(v) for k, v in p})
        except:
            pass
    df = pd.DataFrame(res[1:])

    return df


if __name__ == '__main__':
    # plt.rc('text', usetex=True)
    networks = ['N1b', 'N3b_[M+H]', 'N7b_[M+H]']
    # networks = ['N2', 'N7d', 'N10']
    titles = ['m/z only', 'm/z + in silico CCS', 'm/z + experimental CCS']
    # titles = ['in silico data only', 'floating VAE', 'experimental data only']

    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=600, sharey='row', sharex='col')
    for col, n in enumerate(networks):
        df = load_log(join('../result', n))

        for row, k in enumerate(['val_loss', 'val_decoder_loss', 'val_property_predictor_loss']):
            # plot
            plot(df, key=k, ax=ax[row, col],
                 logx=True, logy=False)
            if (row < 2) or (col != 1):
                ax[row, col].set_xlabel('')
            if col > 0:
                ax[row, col].set_ylabel('')
            if row == 0 and col == 0:
                ax[row, col].set_title(titles[0], weight='bold')
            elif row == 0 and col == 1:
                ax[row, col].set_title(titles[1], weight='bold')
            elif row == 0 and col == 2:
                ax[row, col].set_title(titles[2], weight='bold')

    ax[0, 2].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('../result/figures/learing_curves.png')
