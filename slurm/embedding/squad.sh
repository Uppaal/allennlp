#!/bin/bash
#
#SBATCH --mem=80000
#SBATCH --job-name=s-emb-results
#SBATCH --partition=m40-long
#SBATCH --output=squad-embedding-results-%A.out
#SBATCH --error=squad-embedding-results-%A.err
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

# Train
#python -m allennlp.run train training_config/bidaf_squad.json -s output/squad_emb

# Evaluation on SQuAD
python -m allennlp.run evaluate output/squad_emb/model.tar.gz --evaluation-data-file data/squad/dev-v1.1.json

# Evaluation on NewsQA
python -m allennlp.run evaluate output/squad_emb/model.tar.gz --evaluation-data-file data/NewsQA/test-v1.1.json


# Change:
# - output/PATH
# - job name
# - out and err paths
# - config file
