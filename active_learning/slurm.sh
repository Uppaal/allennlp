#!/bin/bash
#
#SBATCH --mem=30000
#SBATCH --job-name=1-gpu-bidaf-pytorch
#SBATCH --partition=m40-long
#SBATCH --output=batch5_restart_bidaf-pytorch-%A.out
#SBATCH --error=batch5_restart_bidaf-pytorch-%A.err
#SBATCH --gres=gpu:1


# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

python process_logits.py -s ../active_data/newsqa/train_dump.json -feat_label_p ../active_data/newsqa_files/classifier_input.p  -t ../active_data/newsqa_files/classifier_top10.json
#python process_logits.py  -c_i ../active_data/newsqa_files/context_id_combine.p  -logits ../active_data/newsqa_files/combined_logits_newsqa.p  --other_features_file ../active_data/newsqa_files/newsqa_merged.csv -feat_label_p ../active_data/newsqa_filesclassifier_input.p
