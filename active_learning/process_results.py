import argparse
import json
import pickle
from torch.autograd import Variable
import os

from utils import *


def get_args():
    parser = argparse.ArgumentParser()

    home = "./"

    results_file = "data/combined_results.txt"
    data_file = "data/newsqa/dev_dump.json"

    target_file = "data/newsqa/top_dev_dump.json"

    parser.add_argument('-r', "--results_file", default=home + results_file)
    parser.add_argument('-d', "--data_file", default=home + data_file)
    parser.add_argument('-t', "--target_file", default=home + target_file)

    return parser.parse_args()


def load_data_file(data_file):
    fp = open(data_file, 'r')
    data = json.load(fp)
    data = data['data']
    return data


# Construct a dataframe row for analysis,
# Currently computes:
def get_feature_row(context, is_correct, gold_labels, pred_labels):
    split_predicted_answer = context.split()[pred_labels[0]:pred_labels[1]]
    split_predicted_answer = " ".join(split_predicted_answer)
    print("Predicted answer: ",split_predicted_answer)

    split_gold_answer = context.split()[gold_labels[0]:gold_labels[1]]
    split_gold_answer = " ".join(split_gold_answer)
    print("Gold Answer: ", split_gold_answer)


def process(args):
    df_logits = pickle.load(open(args.results_file, 'rb'))
    df_logits = df_logits.set_index('id')
    # print(df_logits)
    # print(df_logits.columns)

    data = load_data_file(args.data_file)

    unavailable_count = 0
    for data_item in data:
        paragraphs = data_item['paragraphs']
        for p in paragraphs:
            context = p['context']
            qas = p['qas']
            for q in qas:
                id = q['id']

                if id not in df_logits.index:
                    unavailable_count += 1
                else:
                    row = df_logits.loc[id]

                    is_correct = row['correct']  # type: int.64
                    gold_labels = row['gold_labels']  # type: list
                    pred_labels = row['predicted_spans']  # type: list

                    dataframe_row = get_feature_row(context, is_correct, gold_labels, pred_labels)

    print(unavailable_count)

def main():
    args = get_args()
    process(args)


if __name__ == "__main__":
    print(os.getcwd())
    main()