import pandas as pd
from process_logits import process
import argparse
from torch.autograd import Variable
import torch
import numpy as np
import pickle

def get_args():
    parser = argparse.ArgumentParser()

    home = "/home/ishita/Downloads/maluuba/allennlp/"
    #source- NewsQA-- on which evaluation is to be done
    parser.add_argument('-s', "--source_file", default='')

    #Need logits to get best span for every id
    parser.add_argument('-t_logits', "--target_logits", default='')
    # dump is the final output dump file
    parser.add_argument('-t_dump', "--target_dump_file", default='')

    parser.add_argument("--score-type", default=0)
    parser.add_argument("--scoring", default='entropy')
    parser.add_argument('-g', "--gpu", default=0)
    parser.add_argument('-p', "--percent", default=1)

    
    
    return parser.parse_args()


def get_top_start_end(span_start_logits,span_end_logits):
    passage_length = len(span_start_logits)
    max_span_log_prob = -1e20 
    span_start_argmax = 0
    for j in range(passage_length):
        val1 = span_start_logits[span_start_argmax]
        if val1 < span_start_logits[j]:
            span_start_argmax = j
            val1 = span_start_logits[j]

        val2 = span_end_logits[j]
        if val1 + val2 > max_span_log_prob:
            best_span_start = span_start_argmax
            best_span_end= j
            max_span_log_prob = val1 + val2
    return best_span_start,best_span_end

def get_id_best_span(logits_file):
    f = open(logits_file,'rb')
    df = pd.read_pickle(f)
    # batch_size = 1
    idx = list(df.id)
    span_start_logits = list(df.span_start_logits)
    span_end_logits = list(df.span_end_logits)
    best_span_start,best_span_end=[],[]
    for i in range(0,len(df)):
        start,end = get_top_start_end(span_start_logits[i],span_end_logits[i]) 
        best_span_start.append(start)
        best_span_end.append(end)
    print(len(idx),len(best_span_end),len(best_span_end	))
    id_best_span = pd.DataFrame({'ids':idx,'best_span_start': best_span_start, 'best_span_end': best_span_end})
    return id_best_span
    
def get_sortedids_score(args):
    top_ids,sorted_ids,score_mul = process(args, int(args.score_type),args.scoring)
    print(len(sorted_ids))
    sorted_ids_score = pd.DataFrame({'ids':sorted_ids})
    return sorted_ids_score

def main():
	args = get_args()
	id_best_span = get_id_best_span(args.target_logits)
	print(len(id_best_span))
	print("id_best_span created")
	id_sorted_score = get_sortedids_score(args)
	print('id_sortedids_score created')
	print(len(id_sorted_score))
	combined_output = pd.merge(id_sorted_score,id_best_span,  how='left', on='ids')
	print("Dumping")
	print(len(combined_output))
	with open(args.target_dump_file, 'wb') as fp:
		pickle.dump(combined_output, fp)

if __name__=='__main__':
	main()