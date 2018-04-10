import numpy as np
import torch


def calculate_total(start, end, dtype):
    start = start.type(dtype)
    end = end.type(dtype)
    s_p = torch.nn.functional.softmax(torch.autograd.Variable(start), dim=0).data
    e_p = torch.nn.functional.softmax(torch.autograd.Variable(end), dim=1).data
    score_mul = s_p * e_p
    score_mul = torch.triu(score_mul)
    score_sum = torch.sum(score_mul)
    score_mul = score_mul / score_sum
    score_mul[score_mul == 0] = 1
    y = torch.log(score_mul)
    y = score_mul * y
    # print("total ")
    total = -1 * torch.sum(y)
    # print(total)
    return total
