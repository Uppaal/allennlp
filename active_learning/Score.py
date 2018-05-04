import numpy as np
import torch
from sklearn.metrics import f1_score
import LogitsScore, SoftmaxScore

def score_using_common(sen1,sen2):
    sen1 = sen1.split()
    sen2 = sen2.split()
    count = len(list(set(sen1).intersection(sen2)))
    return count

def score_f1(true_ans,ans):
    true = [True]*max(len(true_ans),len(ans))
    actual = [ans[i] in true_ans for i in range(len(ans))]
    if len(actual) < len(true):
        actual = actual+[False]*(len(true)-len(actual))
    f1_measure = f1_score(true,actual) 
    # precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
    # recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
    # f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
    return f1_measure

def score_all_using_logits_sum_all(start, end, dtype):
    start, end = score_all(start, end)
    print("Score all usign logits")
    return LogitsScore.calculate_total(start, end, dtype)

def score_all_using_logits_contrast(start, end, dtype):
    start, end = score_all(start, end)
    print("in here")

    return LogitsScore.calculate_diff(start, end, dtype)

def score_topk_using_logits(start, end, dtype, top=1):
    start, end = score_top(start, end, top=top)

    print("Score all using topk")
    return LogitsScore.calculate_total(start, end, dtype)


def score_all_using_softmax_sum_all(start, end, dtype):
    start, end = score_all(start, end)
    print("Score all using softmax")
    return SoftmaxScore.calculate_total(start, end, dtype)

def score_all_using_softmax_contrast(start, end, dtype):
    start, end = score_all(start, end)

    return SoftmaxScore.calculate_contrast(start, end, dtype)

def score_topk_using_softmax(start, end, dtype, top=1):
    start, end = score_top(start, end, top=top)
    print("Score all using topk softm")
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

def score_top2_contrast(start, end, top=1):
    required_len = int(int(top) * len(start) / 100)
    # print("Taking {} scores out of {}".format(required_len, len(start)))

    start = sorted(start)
    start = start[-required_len:]
    start = torch.from_numpy(np.array([start]).transpose())

    end = sorted(end)
    end = end[-required_len:]
    end = torch.from_numpy(np.array([end]))

    return start, end

def score_all_using_classifier(model,X_target):
    result = model.predict(X_target)
    return result    
