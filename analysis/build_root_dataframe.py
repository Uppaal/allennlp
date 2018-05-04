import argparse
import json
import os, nltk

import datefinder
import pandas as pd

from analysis.utils import *

column_names = ['id', 'context-length', 'num-context-ner-features', 'context-ner-features', 'num-context-date-features',
                'context-date-features', 'q-word', 'q-word-pos', 'prev-q-word', 'prev-q-word-pos', 'next-q-word',
                'next-q-word-pos', 'is-q-word-begin', 'num-q-ner-features', 'q-ner-features', 'answer-length',
                'answer-start', 'num-answer-ner-features', 'answer-ner-features', 'num-answer-date-features',
                'answer-date-features']
dataframe = pd.DataFrame(columns=column_names)

dataframe['context-ner-features'] = dataframe['context-ner-features'].astype(object)
dataframe['context-date-features'] = dataframe['context-date-features'].astype(object)
dataframe['q-ner-features'] = dataframe['q-ner-features'].astype(object)
dataframe['answer-ner-features'] = dataframe['answer-ner-features'].astype(object)
dataframe['answer-date-features'] = dataframe['answer-date-features'].astype(object)


def get_args():
    parser = argparse.ArgumentParser()

    home = "."
    # source- NewsQA-- on which evaluation is to be done
    # parser.add_argument('-s', "--source_file", default='')
    # parser.add_argument('-d', "--data", default='data/squad/train-v1.1.json')
    parser.add_argument('-d', "--data", default='data/newsqa/train_dump.json')
    # parser.add_argument('-d', "--data", default='data/newsqa/dev_dump.json')
    # parser.add_argument('-t', "--target", default="data/squad_df.pkl")
    parser.add_argument('-t', "--target", default="data/newsqa_df.pkl")
    # parser.add_argument('-t', "--target", default="data/dev_trial_df.pkl")
    return parser.parse_args()


# returns a list of entities matched
def get_ner_features(line):
    chunked_sentences = ner_tag_line(line)

    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))
    # print(entity_names)
    return entity_names


# returns a list of dates matched
def parse_date_features(line):
    matches = datefinder.find_dates(line, source=True)
    num_dates_matched = list(matches)
    # for date in matches:
    #     print(date)
    return num_dates_matched


def process_context(context):
    tokenized_context_sentences = tokenize_sentences(context)
    context_length = 0
    for sentence in tokenized_context_sentences:
        context_length += len(sentence)

    ner_features = get_ner_features(context)
    date_features = parse_date_features(context)

    context_features = {}
    context_features['context-length'] = context_length
    context_features['ner-features'] = ner_features
    context_features['date-features'] = date_features

    return context_features


def process_question(question):
    question_pos_tags_list = pos_tag_sentences(question)
    question_pos_tags_list = question_pos_tags_list[0]

    ner_features = get_ner_features(question)

    i = 0
    curr = -1
    prev = -1
    next = -1
    for word, tag in question_pos_tags_list:
        if str(tag).startswith("W"):
            # print(word, tag)
            curr = i
            prev = curr - 1
            next = curr + 1
        i += 1

    prev_word = None
    prev_word_pos = None
    if prev != -1:
        prev_word, prev_word_pos = question_pos_tags_list[prev]

    next_word = None
    next_word_pos = None
    if next != len(question_pos_tags_list):
        next_word, next_word_pos = question_pos_tags_list[next]

    curr_word, curr_word_pos = question_pos_tags_list[curr]

    # Denote that the question begins with a question word
    if curr == 0:
        wh_begin = 1
    else:
        wh_begin = 0

    question_features = {}
    question_features['prev-word'] = prev_word
    question_features['prev-word-pos'] = prev_word_pos
    question_features['next-word'] = next_word
    question_features['next-word-pos'] = next_word_pos
    question_features['curr-word'] = curr_word
    question_features['curr-word-pos'] = curr_word_pos
    question_features['wh-begin'] = wh_begin
    question_features['ner-features'] = ner_features

    return question_features


def process_answers_list(answers_list):
    avg_start_sum = 0.0
    avg_answer_length = 0.0
    ner_features = []
    date_features = []
    for answer in answers_list:
        avg_start_sum += answer['answer_start']
        answer_text = answer['text']
        avg_answer_length += len(nltk.word_tokenize(answer_text))
        ner_features.extend(get_ner_features(answer_text))
        date_features.extend(parse_date_features(answer_text))
    avg_start_sum /= len(answers_list)
    avg_answer_length /= len(answers_list)

    answer_features = {}
    answer_features['answer-length'] = avg_answer_length
    answer_features['answer-start'] = avg_start_sum
    answer_features['ner-features'] = ner_features
    answer_features['date-features'] = date_features

    return answer_features


def insert_values_into_dataframe(loc, id, context_features, question_features, answer_features):
    context_length = context_features['context-length']
    context_ner_features = context_features['ner-features']
    num_context_ner_features = len(context_ner_features)
    context_date_features = context_features['date-features']
    num_context_date_features = len(context_date_features)

    answer_length = answer_features['answer-length']
    answer_start = answer_features['answer-start']
    answer_ner_features = answer_features['ner-features']
    num_answer_ner_features = len(answer_ner_features)
    answer_date_features = answer_features['date-features']
    num_answer_date_features = len(answer_date_features)

    prev_word = question_features['prev-word']
    prev_word_pos = question_features['prev-word-pos']
    next_word = question_features['next-word']
    next_word_pos = question_features['next-word-pos']
    curr_word = question_features['curr-word']
    curr_word_pos = question_features['curr-word-pos']
    wh_begin = question_features['wh-begin']
    question_ner_features = question_features['ner-features']
    num_question_ner_features = len(question_ner_features)

    row_values = [id, context_length, num_context_ner_features, context_ner_features, num_context_date_features,
                  context_date_features, curr_word, curr_word_pos, prev_word, prev_word_pos, next_word,
                  next_word_pos, wh_begin, num_question_ner_features, question_ner_features, answer_length,
                  answer_start, num_answer_ner_features, answer_ner_features, num_answer_date_features,
                  answer_date_features]

    dataframe.loc[loc] = row_values


def analyze_dataset():
    args = get_args()

    data = get_json_data(args)
    # print(len(data))

    limit = 2
    count = 0

    loc = 1
    par_index = 1
    print(len(data))
    for item in data:
        paragraphs_list = item['paragraphs']
        # print(len(paragraphs_list))
        print("Par: ", par_index)
        par_index += 1

        for paragraph in paragraphs_list:
            # print(paragraph.keys())
            context = paragraph['context']
            if count < limit:
                count += 1
            else:
                # break
                count += 1
                # pass
            context_features = process_context(context)

            qas_list = paragraph['qas']
            # print(len(qas_list))
            for qas in qas_list:
                id = qas['id']

                question = qas['question']
                question_features = process_question(question)

                answers_list = qas['answers']
                answer_features = process_answers_list(answers_list)

                insert_values_into_dataframe(loc, id, context_features, question_features, answer_features)
                loc += 1
    # print(dataframe.iloc[0])
    # print(dataframe)
    dataframe.to_pickle(args.target)
    df = pd.read_pickle(args.target)
    # print(df)


def get_json_data(args):
    file = open(args.data, 'r')
    data_file = json.load(file)
    data = data_file['data']
    file.close()
    return data


if __name__ == '__main__':
    print(os.getcwd())
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    analyze_dataset()
