#!/bin/bash
#
#SBATCH --mem=50000
#SBATCH --job-name=cl10i1
#SBATCH --partition=m40-long
#SBATCH --output=cl10i1-test-%A.out
#SBATCH --error=cl10i1-test-%A.err
#SBATCH --gres=gpu:1

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

# Test on NewsQA
python -m allennlp.run evaluate output_active/certain-logits-10/iteration_1/model.tar.gz --evaluation-data-file "data/NewsQA/test-v1.1.json"
