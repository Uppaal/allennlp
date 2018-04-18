import argparse
import json
import pickle
from functools import reduce
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from sklearn.linear_model import LinearRegression, SGDRegressor

import LogitsScore, SoftmaxScore, Score
from process_candidate_answers import process_saved_file


def get_args():
    parser = argparse.ArgumentParser()

    home = "/home/ishita/Downloads/maluuba/allennlp/"

    source_logits_file = "active_data/squad_small/combined.p"
    source_file = "active_data/squad/train-v1.1.json"
    target_file = "data/newsqa/top_dev_dump.json"

    parser.add_argument('-s_dev', "--source_logits_file", default=home + source_logits_file)
    parser.add_argument('-t', "--target_file", default=home + target_file)
    parser.add_argument('-s', "--source_file", default=home + source_file)
    parser.add_argument('-logits', "--logits", default=home+"active_data/squad_small/combined_logits.p")
    parser.add_argument('-p', "--percent", default=1)
    parser.add_argument('-g', "--gpu", default=0)
    parser.add_argument('-c_i', "--context_id_combine", default=home+"active_data/context_id_combine.p")
    parser.add_argument('-other_features_file', "--other_features_file", default=home+"active_data/squad_small/squad_merged.csv")
    parser.add_argument('-feature_label_file', "--feature_label_file", default=home+"active_data/squad_small/classifier_input.csv")
    parser.add_argument('-feat_label_p', "--feature_label_file_pickle", default=home+"active_data/squad_small/classifier_input.p")

    '''
    Use 0 for logits
    Use 1 for softmax
    '''
    parser.add_argument("--score-type", default=0)
    parser.add_argument("--k", default=2)

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
            scores = Score.score_all_using_logits_sum_all(start_spans[i], end_spans[i], dtype)
        if score_type == 1:
            scores = Score.score_all_using_logits_contrast(start_spans[i], end_spans[i], dtype)
        elif score_type == 2:
            scores = Score.score_all_using_softmax(start_spans[i], end_spans[i], dtype)
        elif score_type == 3:
            scores = Score.score_topk_using_logits(start_spans[i], end_spans[i], dtype, args.k)
        elif score_type == 4:
            scores = Score.score_topk_using_softmax(start_spans[i], end_spans[i], dtype, args.k)
        entropy_scores.append(scores)

    print("Sorting now")

    score_df = pd.DataFrame(list(zip(logits_ids, entropy_scores)), columns=['ids', 'entropy'])
    # print(df)

    sorted_values = score_df.sort_values(by = 'entropy', ascending = False)
    # print(sorted_values)

    num_top_values = int(int(args.percent) * len(sorted_values) / 100)
    # print(idx)

    top_ids = sorted_values[:num_top_values]
    # top_ids = sorted_values[:num_top_values].ids
    # print(top_ids)
    return top_ids

def get_top_start_end(span_start_logits,span_end_logits,batch_size):
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    max_span_log_prob = [-1e20] * batch_size
    span_start_argmax = [0] * batch_size
    best_word_span = Variable(span_start_logits.data.new()
                              .resize_(batch_size, 2).fill_(0)).long()

    span_start_logits = span_start_logits.data.cpu().numpy()
    span_end_logits = span_end_logits.data.cpu().numpy()

    for b in range(batch_size):  # pylint: disable=invalid-name
        for j in range(passage_length):
            val1 = span_start_logits[b, span_start_argmax[b]]
            if val1 < span_start_logits[b, j]:
                span_start_argmax[b] = j
                val1 = span_start_logits[b, j]

            val2 = span_end_logits[b, j]

            if val1 + val2 > max_span_log_prob[b]:
                best_word_span[b, 0] = span_start_argmax[b]
                best_word_span[b, 1] = j
                max_span_log_prob[b] = val1 + val2
    return best_word_span

def retrieve_sentence(best_span,offset,context):
    start, end = int(best_span[0]),int(best_span[1])
    context = context.split()
    ans = context[start:end]
    ans = " ".join(ans)
    if start-offset <0:
        start=offset
    if end+offset>len(context):
        end = len(context)-offset
    prefix = context[start-offset:start]
    prefix = " ".join(prefix)
    suffix = context[end:end+offset]
    suffix = " ".join(suffix)

    if  '.' in prefix:
        prefix = prefix.split('.')[-1]
    if '.' in suffix:
        suffix = suffix.split('.')[0]        
    return prefix+ans+suffix, ans

def get_feature(df,idx):
    f = df[df.ids==idx]
    return list(df[df.ids==idx].values[0][1:])

def create_ans_ques_feature(args):
    context = pickle.load(open(args.context_id_combine,'rb'))
    
    offset = 5
    print(args.logits)
    file = open(args.logits,'rb')
    df = pd.read_pickle(file)
    batch_size = 1
    idx = list(df.id)
    idxcopy = list(df.id) 
    span_start_logits = list(df.span_start_logits)
    span_end_logits = list(df.span_end_logits)
    flag = 0

    for i in range(0,len(df),batch_size):
        if len(idx[i])<24:
            idxcopy.remove(idx[i])
            continue
        start = Variable(torch.Tensor(span_start_logits[i:i+batch_size]))
        end = Variable(torch.Tensor(span_end_logits[i:i+batch_size]))
        if flag==0:
            flag = 1
            best_span =get_top_start_end(start,end,batch_size)
        else:
            best_span = torch.cat((best_span,get_top_start_end(start,end,batch_size)),0)
        # if i==700:
        #     break
    ids,features,label=[],[],[]
    other_features_file = open(args.other_features_file,'rb')
   
    print("Making features for Classifier")
    df = pd.read_csv(other_features_file)
    
    for i in range(best_span.size()[0]):
        print(i,best_span.size()[0])
        if idxcopy[i] not in context.keys():
            print("key not found")
        else:
            # print(int(best_span[i][0]))
            sentence, true_ans = retrieve_sentence(best_span[i],offset,context[idxcopy[i]]['context'])
            ques = context[idxcopy[i]]['ques']
            ans = context[idxcopy[i]]['ans']
            score = Score.score_using_common(ques,sentence)
            f1 = Score.score_f1(true_ans,ans)
            other_features = get_feature(df,idxcopy[i])
            features.append(other_features+[score])
            ids.append(idxcopy[i])
            label.append(f1)
    
    print("Dumping Features")
    # df_features = pd.DataFrame([ids,features,label],columns=['ids','features','label'])
    # df_features.to_csv(args.feature_label_file, encoding='utf-8', index=False)

    with open(args.feature_label_file_pickle, 'wb') as fp:
        pickle.dump([ids,features,label], fp)
    
    
def create_context_id_pair(args):
    print("Making context-id pair")
    file = open(args.source_file, 'rb')
    f = json.load(file)

    data = f['data']
    new_data = []
    paragraph = {}
    for item in data:
        paragraphs = item['paragraphs']
        for p in paragraphs:
            context = p['context']
            qas = p['qas']
            for q in qas:
                id = q['id']
                ques = q['question']
                ans = q['answers'][0]['text'] 
                paragraph[id]= {'context':context,'ques':ques,'ans':ans}
    print("Dumping context id pair")
    print(args.context_id_combine)
    with open(args.context_id_combine, 'wb') as fp:
        pickle.dump(paragraph, fp)
    # return paragraph
    
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

def training_classifier(args):
    print("Loading Features")
    f = open(args.feature_label_file_pickle,'rb')
    ids,x,y = pickle.load(f)  
    print(np.array(x).shape)
    print(np.array(y).shape)
    linreg = LinearRegression()
    linreg.fit(x,y)
    filename = './trained_model/classifier.sav'
    pickle.dump(linreg, open(filename, 'wb'))

def retrieve_target_ids(args,k=1):
    filename = './trained_model/classifier.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    classifier_input = open('classifier_input.p','rb')
    ids,X,y = pickle.load(open(classifier_input))
    result = loaded_model.predict(X)
    sorted_values = np.array(result).argsort()
    k = int(k*len(sorted_values))
    top_k = sorted_values[:k]
    create_dataset_from_ids(top_k,args)
    print("Done")
    
 
def main():
    args = get_args()
    #create_context_id_pair(args)
    #create_ans_ques_feature(args)
    #training_classifier(args)
    retrieve_target_ids(args,10)
    create_classifier_topk 
    # top_ids = process(args, args.score_type)

    # create_dataset_from_ids(top_ids.ids, args)

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
    main()
    # create_candidate_spans()
    # args = get_args()
