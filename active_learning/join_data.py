import argparse
import json
import os


def get_args():
    parser = argparse.ArgumentParser()

    home = "./"

    # source_logits_file = "data/combined_logits.p"

    source_file = "data/squad/train-v1.1.json"
    al_file = "data/newsqa-1.json"
    target_file = "data/joint-sq-n1.json"

    parser.add_argument('-a', "--al_file", default=home + al_file)
    parser.add_argument('-t', "--target_file", default=home + target_file)
    parser.add_argument('-s', "--source_file", default=home + source_file)


    return parser.parse_args()

def join(args):
    print("Creating file")

    file = open(args.source_file, 'rb')
    f = json.load(file)
    data = f['data']

    al_file = open(args.al_file, 'rb')
    alf = json.load(al_file)
    al_data = alf['data']

    new_data = []

    print("Going to join now ...")
    print(f.keys(), alf.keys())
    print(len(f['data']), len(alf['data']))
    # print(data[0].keys(), al_data[0].keys())
    print(len(new_data))
    new_data.extend(data)
    print(len(new_data))
    new_data.extend(al_data)
    print(len(new_data))

    jsonfinal = {"data": new_data, "version": "4"}

    with open(args.target_file, 'w') as fp:
        json.dump(jsonfinal, fp)


def check(target_file, al_file):
    file = open(target_file, 'rb')
    f = json.load(file)
    data = f['data']
    print(len(data))

    al_file = open(args.al_file, 'rb')
    alf = json.load(al_file)
    al_data = alf['data']
    print(len(al_data))

    count = 0
    for alitem in al_data:
        alparagraphs = alitem['paragraphs']
        for alp in alparagraphs:
            paragraph = []
            alcontext = alp['context']
            alqas = alp['qas']
            qas_new = []
            for alq in alqas:
                alid = alq['id']

                for item in data:
                    paragraphs = item['paragraphs']
                    for p in paragraphs:
                        paragraph = []
                        context = p['context']
                        qas = p['qas']
                        qas_new = []
                        for q in qas:
                            id = q['id']
                            if id == alid:
                                count += 1
                                # print("FOUND!!!!")
    print(count)

if __name__=='__main__':
    print(os.getcwd())
    args = get_args()
    join(args=args)
    check(args.target_file, args.al_file)
