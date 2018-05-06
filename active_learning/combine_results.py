import pandas as pd
import os
import pickle
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    s_dir = "data/results_dump"
    o_file = "data/combined_results.txt"
    parser.add_argument('-s_dir', default=s_dir)
    parser.add_argument('-o_file', default=o_file)

    return parser.parse_args()


def combine_logits_from_file(args):
    directory = args.s_dir

    print("\nCreate logits file from ", directory)
    final = []
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                file = open(os.path.join(subdir, filename), 'r')
                file = file.read()
                file = file.split('][')
                for i in range(len(file)):
                    ele = file[i]
                    if i != len(file) - 1:
                        ele = ele + ']'
                    if i != 0:
                        ele = '[' + ele
                    df = pd.read_json(ele)
                    final.append(df)
    result = pd.concat(final)
    f = open(args.o_file, 'wb')
    pickle.dump(result, f)


if __name__ == '__main__':
    print(os.getcwd())
    args = get_args()
    combine_logits_from_file(args)
