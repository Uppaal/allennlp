import numpy as np
import torch

import LogitsScore, SoftmaxScore


def score_all_using_logits(start, end, dtype):
    start, end = score_all(start, end)

    return LogitsScore.calculate_total(start, end, dtype)


def score_topk_using_logits(start, end, dtype, top=1):
    start, end = score_top(start, end, top=top)

    return LogitsScore.calculate_total(start, end, dtype)


def score_all_using_softmax(start, end, dtype):
    start, end = score_all(start, end)

    return SoftmaxScore.calculate_total(start, end, dtype)


def score_topk_using_softmax(start, end, dtype, top=1):
    start, end = score_top(start, end, top=top)

    return SoftmaxScore.calculate_total(start, end, dtype)


def score_all(start, end):
    start = torch.from_numpy(np.array([start]).transpose())
    end = torch.from_numpy(np.array([end]))

    return start, end


def score_top(start, end, top=1):
    required_len = int(int(top) * len(start) / 100)
    # print("Taking {} scores out of {}".format(required_len, len(start)))

    start = sorted(start)
    start = start[-required_len:]
    start = torch.from_numpy(np.array([start]).transpose())

    end = sorted(end)
    end = end[-required_len:]
    end = torch.from_numpy(np.array([end]))

    return start, end
