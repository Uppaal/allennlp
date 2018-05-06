#!/bin/bash
#
#SBATCH --mem=50000
#SBATCH --job-name=s-emb
#SBATCH --partition=m40-long
#SBATCH --output=squad-train-emb-%A.out
#SBATCH --error=squad-train-emb-%A.err
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

cd /home/usaxena/work/s18/696/allen2/

# Train on squad
python -m allennlp.run train training_config/bidaf_embedding.json -s output/squad_emb

# Test on SQuAD
python -m allennlp.run evaluate output/squad_emb/model.tar.gz --evaluation-data-file "data/squad/dev-v1.1.json"

# Test on NewsQA
python -m allennlp.run evaluate output/squad_emb/model.tar.gz --evaluation-data-file "data/NewsQA/test-v1.1.json"
