import argparse
import json
import pickle
from functools import reduce
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.linear_model import LinearRegression, SGDRegressor
import os

import LogitsScore, SoftmaxScore, Score
from process_candidate_answers import process_saved_file
from utils import *

def get_args():
    parser = argparse.ArgumentParser()

    home = "./"

    source_logits_file = "active_data/squad_small/combined.p"
    source_file = "active_data/squad/train-v1.1.json"
    target_file = "data/newsqa/top_dev_dump.json"
  
    parser.add_argument('-s', "--source_file", default=home + source_file)
    parser.add_argument('-t', "--target_file", default=home + target_file)
    parser.add_argument('-t_dump', "--target_dump_file", default=home + target_file)


    parser.add_argument('-t_logits', "--target_logits", default=home+"active_data/squad_small/combined_logits.p")

    # dumping data
    parser.add_argument('-t_selected_ids', "--target_dump_selected_ids", default=home + target_dump_selected_ids)
    parser.add_argument('-t_unselected_ids', "--target_dump_unselected_ids", default=home + target_dump_unselected_ids)

    parser.add_argument('-p', "--percent", default=1)
    parser.add_argument('-g', "--gpu", default=0)

#classifier arguments
    parser.add_argument('-s_logits', "--source_logits", default=home+"active_data/squad_small/combined_logits.p")
    parser.add_argument('-s_other_features_file', "--source_other_features_file", default=home+"active_data/squad_small/squad_merged.csv")
    parser.add_argument('-t_other_features_file', "--target_other_features_file", default=home+"active_data/squad_small/squad_merged.csv")
    parser.add_argument('-feature_label_file', "--feature_label_file", default=home+"active_data/squad_small/classifier_input.csv")
    parser.add_argument('-feat_label_p', "--feature_label_file_pickle", default=home+"active_data/squad_small/classifier_input.p")

    '''
    Use 0 for logits
    Use 1 for softmax
    '''
    parser.add_argument("--score-type", default=0)
    parser.add_argument("--scoring", default='entropy')
    parser.add_argument("--k", default=2)

    return parser.parse_args()


def process(args, score_type, scoring):
    df_logits = pickle.load(open(args.target_logits, 'rb'))
    logits_ids = list(df_logits.id)
    start_spans = list(df_logits.span_start_logits)
    end_spans = list(df_logits.span_end_logits)
    entropy_scores = []

    num_datapoints = len(start_spans)
    print("Number of datapoints: {}".format(num_datapoints))
    score_mul_list=[]
    dtype = torch.cuda.FloatTensor
    if (args.gpu == 0):
        dtype = torch.FloatTensor

        if scoring == 'entropy':
            for i in range(len(start_spans)):
                # for i in range(50):
                # print(i)
                scores = None
                # print(type(start_spans),type(end_spans))
                # print(end_spans[i])
                if score_type == 0:
                    scores, score_mul = Score.score_all_using_logits_sum_all(start_spans[i], end_spans[i], dtype)
                elif score_type == 1:
                    scores, score_mul = Score.score_all_using_logits_contrast(start_spans[i], end_spans[i], dtype)
                elif score_type == 2:
                    scores, score_mul = Score.score_all_using_softmax_sum_all(start_spans[i], end_spans[i], dtype)
                elif score_type == 3:
                    scores, score_mul = Score.score_all_using_softmax_contrast(start_spans[i], end_spans[i], dtype)
                elif score_type == 4:
                    scores, score_mul = Score.score_topk_using_logits(start_spans[i], end_spans[i], dtype, args.k)
                elif score_type == 5:
                    scores, score_mul = Score.score_topk_using_softmax(start_spans[i], end_spans[i], dtype, args.k)
                entropy_scores.append(scores)
                # score_mul_list.append(score_mul)
        if scoring == 'classifier':
            ids, X_source, y_source = create_features(args.source_file, args.source_logits,
                                                      args.source_other_features_file, args, feature_source_label_file)
            model = training_classifier(X_source, y_source)
            ids, X_target, _ = create_features(args.target_file, args.target_logits, args.target_other_features_file,
                                               args, args.feature_target_label_file, flag='target')
            scores = Score.score_all_using_classifier(model, X_target)

    print("Sorting now")
    if scoring =='classifier':
        sorted_values = np.array(scores).argsort()
    if scoring =='entropy':
        score_df = pd.DataFrame(list(zip(logits_ids, entropy_scores)), columns=['ids', 'entropy'])
        sorted_values = score_df.sort_values(by = 'entropy', ascending = True)

    sorted_values.to_csv(args.target_dump_selected_ids.split(".")[0][:] + ".csv", encoding='utf-8', index=False)

    num_top_values = int(int(args.percent) * len(sorted_values) / 100)

    top_ids = list(sorted_values[:num_top_values].ids)
    return top_ids,sorted_values.ids,score_mul_list


def create_dataset_from_ids(ids, args):
    print("Creating file")
    ids = list(ids)
    print("inside create data")
    print(len(ids))
    file = open(args.target_file, 'rb')
    print("Source File is ")
    print(args.source_file)
    f = json.load(file)
    data = f['data']
    data_in_ids = []
    data_not_in_ids=[]
    for item in data:
        paragraphs = item['paragraphs']
        for p in paragraphs:
            paragraph_in_ids = []
            paragraph_not_in_ids = []
            context = p['context']
            qas = p['qas']
            qas_in_ids= []
            qas_not_in_ids= []
            for q in qas:
                id = q['id']
                if id in ids:
                    # print("FOUND IT")
                    print(q['id'])
                    qas_in_ids.append(q)
                else:
                    qas_not_in_ids.append(q)
            if len(qas_in_ids) != 0:
                paragraph_in_ids.append({"context": context, "qas": qas_in_ids})
            if len(qas_not_in_ids) != 0:
                paragraph_not_in_ids.append({"context": context, "qas": qas_not_in_ids})
            if len(paragraph_in_ids) != 0:
                data_in_ids.append({"paragraphs": paragraph_in_ids, "title": "Hello"})
            if len(paragraph_not_in_ids) != 0:
                data_not_in_ids.append({"paragraphs": paragraph_not_in_ids, "title": "Hello"})

    jsonfinal_in_ids = {"data": data_in_ids, "version": "4"}
    jsonfinal_not_in_ids = {"data": data_not_in_ids, "version": "4"}

    with open(args.target_dump_selected_ids, 'w') as fp:
        json.dump(jsonfinal_in_ids, fp)
    with open(args.target_dump_unselected_ids, 'w') as fp:
        json.dump(jsonfinal_not_in_ids, fp)


def main():
    args = get_args()
    top_ids,_,_ = process(args, int(args.score_type),args.scoring)
    #retrieve_target_ids(args,10)
    #create_classifier_topk
    print(len(top_ids))
    print(top_ids)
    #input()
    create_dataset_from_ids(top_ids,args)

def create_classifier_topk(args):
    classifier_input = open(args.feature_label_file_pickle,'rb')
    ids,X,y = pickle.load(classifier_input)
    result = loaded_model.predict(X)
    sorted_values = np.array(result).argsort()
    k = int(k*len(sorted_values)/100)
    top_k =sorted_values[:k]
    ids = np.array(ids)
    ids_req = ids[top_k]
    return ids_req

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
    # create_candidate_spans()
    # args = get_args()
