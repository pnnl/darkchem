import pandas as pd
import argparse
import numpy as np
import os


if __name__ == '__main__':
    # initialize parser
    parser = argparse.ArgumentParser(description='Split large tsv into smaller files.')

    # inputs/outputs
    parser.add_argument('data', type=str, help='Path to input .tsv file.')
    parser.add_argument('output', type=str, help='Path to output folder (str).')
    parser.add_argument('--every', '-n', type=int, default=1000000,
                        help='Number of entries per file (str, default=1000000).')
    parser.add_argument('--shuffle', '-s', action='store_true', default=False,
                        help='Shuffle input (optional).')

    # parse args
    args = parser.parse_args()

    # read
    df = pd.read_csv(args.data, sep='\t')

    # shuffle
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # prepare output
    name = os.path.splitext(os.path.basename(args.data))[0]
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # chunk
    for i, (g, chunk) in enumerate(df.groupby(np.arange(len(df.index)) // args.every)):
        chunk.to_csv(os.path.join(args.output, '%s_%03d.tsv' % (name, i)), sep='\t', index=None)
