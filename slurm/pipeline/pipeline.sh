#!/bin/bash
#
#SBATCH --mem=50000
#SBATCH --job-name=pipeline
#SBATCH --partition=titanx-long
#SBATCH --output=pipeline-%A.out
#SBATCH --error=pipeline-%A.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruppaal@cs.umass.edu

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python/3.6.1
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44

## Change this line so that it points to your bidaf github folder
cd ../..

# Change active learning samples into correct format (Done manually, using something along these lines):
# mkdir classifier_top_10
# mv classifier_top_10.json classifier_top_10/
# cd classifier_top_10
# cp classifier_top_10.json train.json
# Create dataset for joint training,
# python joint_train.py --target_sampling_ratio 0.1 --debug_ratio 1.0 --train_ratio 0.9 --source_dir "data/squad/" --target_dir "data/active_learning/softmax/10/" --output_dir "data/active_learning/softmax/10/split_0.1/"



cp output/squad/model.tar.gz output_active/iteration_0/model.tar.gz

#for i in {1..5};
#do

# 1. Evaluate on NewsQA
# python -m allennlp.run evaluate output_active/iteration_0/model.tar.gz --evaluation-data-file "data/NewsQA/train-v1.1.json"
python -m allennlp.run evaluate output_active/iteration_0/model.tar.gz --evaluation-data-file "data/NewsQA/dev-v1.1.json"

# 2. Select top k%
cd active_learning
python combine_logits.py 
# python process_logits.py --percent 1 --gpu 1 --source_logits_file "data/logits.p" --source_file "../data/NewsQA/train-v1.1.json" --target_file "data/logits_train_dump_1.json"
python process_logits.py --percent 1 --gpu 1 --source_logits_file "data/logits.p" --source_file "../data/NewsQA/dev-v1.1.json" --target_file "data/logits_dev_dump_1.json"

# 3. Joint selected data with SQuAD
cp logits_train_dump_1.json ../data/active_learning/pipeline/train.json
cd ..
python joint_train.py --target_sampling_ratio 0.0 --debug_ratio 1.0 --train_ratio 0.9 --source_dir "data/squad/" --target_dir "data/active_learning/pipeline/" --output_dir "data/active_learning/pipeline/"
rm train.json

# 4. Finetune on joint
python -m allennlp.run fine-tune \
    --model-archive output_active/iteration_0/model.tar.gz \
    --config-file training_config/bidaf_finetune.json \
    --serialization-dir output_active/iteration_1

#done
