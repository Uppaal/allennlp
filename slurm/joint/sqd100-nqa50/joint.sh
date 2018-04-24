#!/bin/bash
#
#SBATCH --mem=60000
#SBATCH --job-name=j-nqa50-0
#SBATCH --partition=titanx-long
#SBATCH --output=nqa-050-sratio0-bsize15-%A.out
#SBATCH --error=nqa-050-sratio0-bsize15-%A.err
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
#python joint_train.py --target_sampling_ratio 0.1 --debug_ratio 0.5 --train_ratio 0.9 --output_dir "data/joint_nqa_050_0.1/"

# Train
python -m allennlp.run train training_config/bidaf_joint.json -s output/joint_nqa_050_0

# Evaluate on SQuAD
python -m allennlp.run evaluate output/joint_nqa_050_0/model.tar.gz --evaluation-data-file "data/squad/dev-v1.1.json"

# Evaluate on NewsQA
python -m allennlp.run evaluate output/joint_nqa_050_0/model.tar.gz --evaluation-data-file "data/NewsQA/test-v1.1.json"



# Change:
# - output/PATH
# - job name
# - out and err paths
# - config file