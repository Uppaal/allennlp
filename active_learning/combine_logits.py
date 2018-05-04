import pandas as pd
import os
import pickle
import argparse

def combine_logits_from_file(args):
    directory = os.path.join("active_learning","data",args.dir,args.iteration,"eval_per_epoch") #"data/eval_per_epoch"
    print("\nCreate logits file from ",directory)
    final = []
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".json"):
                file = open(os.path.join(subdir, filename), 'r')
                file = file.read()
                file = file.split('][')
                for i in range(len(file)):
                    ele = file[i]
                    if i != len(file) - 1:
                        ele = ele + ']'
                    if i != 0:
                        ele = '[' + ele
                    df = pd.read_json(ele)
                    final.append(df)
    result = pd.concat(final)
    path = os.path.join("active_learning","data", args.dir, args.iteration)
    f = open(path+'/logits.p', 'wb')
    pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dir", default=None)
    parser.add_argument('-i', "--iteration", default=None)
    args = parser.parse_args()
    combine_logits_from_file(args)
