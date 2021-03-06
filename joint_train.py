import os
import pandas as pd
import argparse
import json

def main():
    args = get_args()
    process(args)

def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = "data/squad/"
    target_dir = "data/NewsQA/"
    output_dir = "data/Joint_01/"
   
    # train_ratio: How to split the data into train and dev sets (from the train file). E.g. 0.9 splits to 90% training data and 10% for dev.
    # debug_ratio: What percentage of target directory should be included. E.g. 0.05 means 5% NewsQA is added to 100% SQuAD.
    # target_sampling_ratio: How the target directory should be oversampled to reduce class imbalance between source and target. Do not cross values above 0.3.

    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-o', "--output_dir", default=output_dir)
    
    parser.add_argument("--train_ratio", default=0.9, type=float)
    parser.add_argument("--debug_ratio", default=1.0, type=float)
    parser.add_argument("--data_ratio", default=0.01, type=float)
    parser.add_argument("--target_sampling_ratio", default=0.0, type=float)

    return parser.parse_args()

def process(args):
    source_dir = args.source_dir
    target_dir = args.target_dir
    output_dir = args.output_dir
    sampling_ratio = args.target_sampling_ratio
    data_ratio = args.data_ratio
    train_ratio = args.train_ratio
    debug_ratio = args.debug_ratio
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    splits = ["train","dev"]
    for split in splits:
        fname = split+"-v1.1.json"
        tfname = split+".json" 
        print("Source: ",source_dir+fname)
        print("Target: ", target_dir+tfname)
        sf = pd.read_json(source_dir+fname)
        tf = pd.read_json(target_dir+tfname)

        output_data = []
        output_version = []
        if split == "dev":
            output_data.extend(sf['data'])
            output_data.extend(tf['data'])
            output_version.extend(sf['version'])
            output_version.extend(tf['version'])
            fname = "test-v1.1.json"
        else:
            dev_output_data = []
            dev_output_version = []
            s_len = len(sf['data'])
            t_len = int(len(tf['data'])*debug_ratio)
            output_data.extend(sf['data'][0:int(s_len * train_ratio)])
            dev_output_data.extend(sf['data'][int(s_len * train_ratio):s_len])
            output_version.extend(sf['version'][0:int(s_len * train_ratio)])
            dev_output_version.extend(sf['version'][int(s_len * train_ratio):s_len])
            s_qs = 0
            t_qs = 0
            for i in range(s_len):
                p_len = len(sf['data'][i]['paragraphs'])
                for j in range(p_len):
                    s_qs += len(sf['data'][i]['paragraphs'][j]['qas'])
            for i in range(t_len):
                p_len = len(tf['data'][i]['paragraphs'])
                for j in range(p_len):
                    t_qs += len(tf['data'][i]['paragraphs'][j]['qas'])
            multiplier = 1.0
            r = sampling_ratio
            new_m = r*s_qs/(t_qs*(1-r))
            multiplier = max(multiplier,new_m)
            for i in range(int(multiplier)):
                output_data.extend(tf['data'][0:int(t_len * train_ratio)])
                dev_output_data.extend(tf['data'][int(t_len * train_ratio):t_len])
                output_version.extend(tf['version'][0:int(t_len * train_ratio)])
                dev_output_version.extend(tf['version'][int(t_len * train_ratio):t_len])
            t_len = int(t_len*multiplier - int(multiplier))
            output_data.extend(tf['data'][0:int(t_len * train_ratio)])
            dev_output_data.extend(tf['data'][int(t_len * train_ratio):t_len])
            output_version.extend(tf['version'][0:int(t_len * train_ratio)])
            dev_output_version.extend(tf['version'][int(t_len * train_ratio):t_len])
            dev_of = {"data": dev_output_data, "version": dev_output_version}
            with open(output_dir + "dev-v1.1.json",'w') as fp:
                json.dump(dev_of, fp)
        of = {"data": output_data, "version": output_version}
        with open(output_dir+fname,'w') as fp:
            json.dump(of, fp)

if __name__ =="__main__":
    main()
