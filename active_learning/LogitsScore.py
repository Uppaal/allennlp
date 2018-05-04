import numpy as np
import torch


def get_span_probability(start, end, dtype):
    start = start.type(dtype)
    end = end.type(dtype)
    s_p = start
    e_p = end
    score_mul = s_p * e_p
    score_mul = torch.triu(score_mul)
    score_sum = torch.sum(score_mul)
    score_mul = score_mul / score_sum
    return score_mul

def calculate_total(start,end,dtype):
    score_mul = get_span_probability(start,end,dtype)
    score_mul[score_mul <= 0] = 1
    y = torch.log(score_mul)
    y = score_mul * y
    total = -1 * torch.sum(y)
    return total,score_mul

def calculate_diff(start,end, dtype, k=10):
    score_mul = get_span_probability(start,end,dtype)
    # score_mul[score_mul <= 0] = 0
    y = score_mul.view(-1).nonzero()
    topk,indices = torch.topk(y,k,dim = 0)
    topk_diff = torch.max(topk) - topk
    # print(topk_diff.size())
    assert topk_diff.size()[0] == k
    score = torch.sum(torch.max(topk) - topk )
    return score,score_mul