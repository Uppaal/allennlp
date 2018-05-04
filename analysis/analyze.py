import argparse

import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()

    home = "."
    # source- NewsQA-- on which evaluation is to be done
    # parser.add_argument('-s', "--source_file", default='')

    # parser.add_argument('-d', "--dataframe", default='data/squad_df.pkl')
    # parser.add_argument('-d', "--dataframe", default='data/newsqa_df.pkl')
    parser.add_argument('-d', "--dataframe", default='data/dev_df.pkl')
    # parser.add_argument('-t', "--target", default="data/trial_df.pkl")
    return parser.parse_args()

def analyze():
    args = get_args()

    df = pd.read_pickle(args.dataframe)
    print(df)


if __name__ == '__main__':
    analyze()