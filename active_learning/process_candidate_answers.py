import json
import pickle

import numpy as np


def get_top_scores(start, end, top=1):
    required_len = int(top * len(start) / 100.0)
    # print("Taking {} scores out of {}".format(required_len, len(start)))

    start = np.array(start)
    start = np.argsort(start)

    start = start[-required_len:]

    end = np.array(end)
    end = np.argsort(end)

    end = end[-required_len:]

    # print(start, end)
    return start, end


def build_candidate_answers(context, correct_answer_length, start, end):
    candidate_spans = []
    split_context = context.split(" ")
    context_len = len(split_context)
    for s in start:
        if s - correct_answer_length > 0 and s + correct_answer_length < context_len:
            candidate_spans.append((s - correct_answer_length, s + correct_answer_length))

    for e in end:
        if e - correct_answer_length > 0 and e + correct_answer_length < context_len:
            candidate_spans.append((e - correct_answer_length, e + correct_answer_length))

    # print(candidate_spans)
    candidate_answers = []
    for span in candidate_spans:
        candidate_answer = split_context[span[0]:span[1]]
        # print(candidate_answer)
        candidate_answers.append(candidate_answer)

    return candidate_answers


def get_answer_length(correct_answers):
    l = 0
    for correct_answer in correct_answers:
        if l <= len(correct_answer.split(" ")):
            l = len(correct_answer.split(" "))
    return l


def score_potential_answers(question, correct_answers, candidate_answers):
    question_sum = 0
    answer_sum = 0
    split_question = question.split(" ")
    # print(split_question)
    # print(candidate_answers)
    c3 = [len(list(filter(lambda x: x in split_question, sublist))) for sublist in candidate_answers]

    question_sum = sum(c3)


    correct_answer_split = [answer.split(" ") for answer in correct_answers]
    # print(correct_answer_split)
    for split_answer in correct_answer_split:
        c4 = [len(list(filter(lambda x: x in split_answer, sublist))) for sublist in candidate_answers]
        local_answer_matches = sum(c4)
        answer_sum += local_answer_matches

    # print(question_sum, answer_sum)
    return question_sum, answer_sum


def score_ids(candidate_items):
    break_count = 0
    score_per_id = []
    for item in candidate_items:
        break_count += 1

        id = item[0]
        context = item[1]
        question = item[2]
        correct_answers = item[3]
        start_spans = item[4]
        end_spans = item[5]

        start, end = get_top_scores(start_spans, end_spans)
        correct_answer_length = get_answer_length(correct_answers)
        candidate_answers = build_candidate_answers(context, correct_answer_length, start, end)

        per_id_q_sum, per_id_a_sum = score_potential_answers(question, correct_answers, candidate_answers)
        # if break_count > 2:
        #     break
        score_per_id.append([id, per_id_q_sum, per_id_a_sum])
    return score_per_id


def process_saved_file(args):
    df_logits = pickle.load(open(args.source_logits_file, 'rb'))
    logits_ids = list(df_logits.id)
    start_spans = list(df_logits.span_start_logits)
    end_spans = list(df_logits.span_end_logits)

    print(len(logits_ids))
    fp = open(args.source_file, 'r')
    data = json.load(fp)
    data = data['data']
    count = 0

    candidate_items = []
    for item in data:
        paragraphs = item['paragraphs']
        for p in paragraphs:
            context = p['context']
            qas = p['qas']

            for q in qas:
                # print(qas)
                id = q['id']
                if id in logits_ids:
                    id_index = logits_ids.index(id)

                    count += 1
                    correct_answers = []
                    for potential_answer in q['answers']:
                        correct_answers.append(potential_answer['text'])

                    candidate_items.append(
                        [id, context, q['question'], correct_answers, start_spans[id_index], end_spans[id_index]])
    fp.close()
    print(len(candidate_items))
    # print(candidate_items)

    per_id_sums = score_ids(candidate_items)
    return per_id_sums
