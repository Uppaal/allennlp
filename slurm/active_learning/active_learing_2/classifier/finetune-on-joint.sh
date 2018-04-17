#!/bin/bash
#
#SBATCH --mem=50000
#SBATCH --job-name=a-c10-ft-j
#SBATCH --partition=m40-long
#SBATCH --output=classifier-top10-finetune-on-joint-%A.out
#SBATCH --error=classifier-top10-finetune-on-joint-%A.err
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
cd ../../../..

# Change active learning samples into correct format (Done manually, using something along these lines):
# mkdir classifier_top_10
# mv classifier_top_10.json classifier_top_10/
# cd classifier_top_10
# cp classifier_top_10.json train.json
# Create dataset for joint training,
# python joint_train.py --target_sampling_ratio 0.1 --debug_ratio 1.0 --train_ratio 0.9 --source_dir "data/squad/" --target_dir "data/active_learning/softmax/10/" --output_dir "data/active_learning/softmax/10/split_0.1/"
# Then, rm data/active_learning/active_learning2/classifier_top_10/train.json

# Train
python -m allennlp.run fine-tune \
    --model-archive output/squad/model.tar.gz \
    --config-file training_config/bidaf_active.json \
    --serialization-dir output/active_classifier_top_10_fn-joint

# Evaluate on SQuAD
python -m allennlp.run evaluate output/active_classifier_top_10_fn-joint/model.tar.gz --evaluation-data-file "data/squad/dev-v1.1.json"

# Evaluate on NewsQA
python -m allennlp.run evaluate output/active_classifier_top_10_fn-joint/model.tar.gz --evaluation-data-file "data/NewsQA/test-v1.1.json"



# Change:
# - output/PATH
# - job name
# - out and err paths
# - config file
