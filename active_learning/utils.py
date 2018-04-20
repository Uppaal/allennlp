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

def create_context_id_pair(source_file):
    print("Making context-id pair")
    file = open(source_file, 'rb')
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
    return paragraph

def create_feature(source_file,logits, other_features_file, feature_label_file_pickle,flag='source'):
    context_id_pair = create_context_id_pair(source_file)
    # context = pickle.load(open(args.context_id_combine,'rb'))
    
    offset = 5
    file = open(logits,'rb')
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
    ids,features,label=[],[],[]
    other_features_file = open(other_features_file,'rb')
  
    print("Making features for Classifier")
    df = pd.read_csv(other_features_file)
    
    for i in range(best_span.size()[0]):
        print(i,best_span.size()[0])
        if idxcopy[i] not in context.keys():
            print("key not found")
        else:
            sentence, true_ans = retrieve_sentence(best_span[i],offset,context[idxcopy[i]]['context'])
            ques = context[idxcopy[i]]['ques']
            ans = context[idxcopy[i]]['ans']
            score = Score.score_using_common(ques,sentence)
            other_features = get_feature(df,idxcopy[i])
            features.append(other_features+[score])
            ids.append(idxcopy[i])
            if flag == "source":
                f1 = Score.score_f1(true_ans,ans)
                label.append(f1)
    
    print("Dumping Features")

    with open(feature_label_file_pickle, 'wb') as fp:
        pickle.dump([ids,features,label], fp)
    return([ids,features,label])

def training_classifier(features,labels):
    print(np.array(features).shape)
    print(np.array(labels).shape)
    linreg = LinearRegression()
    linreg.fit(features,labels)
    filename = './trained_model/classifier.sav'
    pickle.dump(linreg, open(filename, 'wb'))
    return linreg