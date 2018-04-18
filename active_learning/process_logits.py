import argparse
import json
import pickle
from functools import reduce

import pandas as pd
import torch
import os

import LogitsScore, SoftmaxScore, Score
from process_candidate_answers import process_saved_file


def get_args():
    parser = argparse.ArgumentParser()

    home = "./"

    source_logits_file = "data/combined_logits.p"
    source_file = "data/newsqa/dev_dump.json"
    target_file = "data/newsqa/top_dev_dump.json"

    parser.add_argument('-s_dev', "--source_logits_file", default=home + source_logits_file)
    parser.add_argument('-t', "--target_file", default=home + target_file)
    parser.add_argument('-s', "--source_file", default=home + source_file)
    parser.add_argument('-p', "--percent", default=1)
    parser.add_argument('-g', "--gpu", default=0)

    '''
    Use 0 for logits
    Use 1 for softmax
    '''
    parser.add_argument("--score-type", default=2)
    parser.add_argument("--k", default=1)


    return parser.parse_args()


def process(args, score_type):
    df_logits = pickle.load(open(args.source_logits_file, 'rb'))
    logits_ids = list(df_logits.id)
    start_spans = list(df_logits.span_start_logits)
    end_spans = list(df_logits.span_end_logits)

    entropy_scores = []

    num_datapoints = len(start_spans)
    print("Number of datapoints: {}".format(num_datapoints))

    dtype = torch.cuda.FloatTensor
    if (args.gpu == 0):
        dtype = torch.FloatTensor

    for i in range(len(start_spans)):
    # for i in range(50):
        scores = None
        if score_type == 0:
            scores = Score.score_all_using_logits(start_spans[i], end_spans[i], dtype)
        elif score_type == 1:
            scores = Score.score_all_using_softmax(start_spans[i], end_spans[i], dtype)
        elif score_type == 2:
            scores = Score.score_topk_using_logits(start_spans[i], end_spans[i], dtype, args.k)
        elif score_type == 3:
            scores = Score.score_topk_using_softmax(start_spans[i], end_spans[i], dtype, args.k)
        entropy_scores.append(scores)

    print("Sorting now")

    score_df = pd.DataFrame(list(zip(logits_ids, entropy_scores)), columns=['ids', 'entropy'])
    # print(df)

    sorted_values = score_df.sort_values('entropy', ascending=False)
    # print(sorted_values)

    num_top_values = int(int(args.percent) * len(sorted_values) / 100)
    # print(idx)

    top_ids = sorted_values[:num_top_values]
    # top_ids = sorted_values[:num_top_values].ids
    # print(top_ids)
    return top_ids


def create_dataset_from_ids(ids, args):
    print("Creating file")
    ids = list(ids)

    file = open(args.source_file, 'rb')
    f = json.load(file)
    data = f['data']
    new_data = []

    for item in data:
        paragraphs = item['paragraphs']
        for p in paragraphs:
            paragraph = []
            context = p['context']
            qas = p['qas']
            qas_new = []
            for q in qas:
                id = q['id']
                if id in ids:
                    # print("FOUND IT")
                    print(q['id'])
                    qas_new.append(q)
            if len(qas_new) != 0:
                paragraph.append({"context": context, "qas": qas_new})
            if len(paragraph) != 0:
                new_data.append({"paragraphs": paragraph, "title": "Hello"})
    jsonfinal = {"data": new_data, "version": "4"}

    with open(args.target_file, 'w') as fp:
        json.dump(jsonfinal, fp)


def main():
    args = get_args()

    top_ids = process(args, args.score_type)

    create_dataset_from_ids(top_ids.ids, args)


def create_candidate_spans():
    args = get_args()
    # print(args.percent)
    args.percent = 100
    top_id_all_logits = process(args, 0)
    top_id_all_softmax = process(args, 1)
    top_id_all_top1_logits = process(args, 2)
    top_id_all_top1_softmax = process(args, 3)
    mergeable_dfs = [top_id_all_logits, top_id_all_softmax, top_id_all_top1_logits, top_id_all_top1_softmax]
    df_entropy = reduce(lambda left,right: pd.merge(left,right,on='ids'), mergeable_dfs)
    # print(df_final)
    # print(df_final.values.tolist())
    df_sum_matches = process_saved_file(args)
    # print(df_sum_matches)
    df_sum_matches = pd.DataFrame(df_sum_matches)
    df_sum_matches.columns = ['ids', 'q_sum', 'a_sum']
    # print(df_sum_matches)
    df_final = df_sum_matches.merge(df_entropy, on='ids')
    df_final.to_csv("data/merged.csv", encoding='utf-8', index=False)



if __name__ == "__main__":
    print(os.getcwd())
    main()
    #create_candidate_spans()
    # args = get_args()
