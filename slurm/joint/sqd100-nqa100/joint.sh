#!/bin/bash
#
#SBATCH --mem=60000
#SBATCH --job-name=jf-10
#SBATCH --partition=m40-long
#SBATCH --output=full-joint-%A.out
#SBATCH --error=full-joint-%A.err
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
cd ../../..

# Create dataset for joint training
#python joint_train.py --target_sampling_ratio 0.1 --debug_ratio 0.01 --train_ratio 0.9 --output_dir "data/joint_nqa_001_0.1/"

# Train
python -m allennlp.run train training_config/bidaf_joint.json -s output/full_joint

# Evaluate on SQuAD
python -m allennlp.run evaluate output/full_joint/model.tar.gz --evaluation-data-file "data/squad/dev-v1.1.json"

# Evaluate on NewsQA
python -m allennlp.run evaluate output/full_joint/model.tar.gz --evaluation-data-file "data/NewsQA/test-v1.1.json"


# Change:
# - output/PATH
# - job name
# - out and err paths
# - config file
