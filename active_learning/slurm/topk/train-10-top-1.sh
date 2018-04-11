#!/bin/bash
#
#SBATCH --mem=80000
#SBATCH --job-name=al-bidaf
#SBATCH --partition=m40-long
#SBATCH --output=l-10-1-al-%A.out
#SBATCH --error=l-10-1-al-%A.err
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

pip install --user torchvision pandas
pip install  --user http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl 

## Change this line so that it points to your bidaf github folder
cd /home/usaxena/work/s18/696/allennlp/active_learning

python process_logits.py --percent 10 --gpu 1 --k 1 --source_logits_file "data/train-logits.p" --source_file "data/newsqa/train_dump.json" --target_file "data/newsqa/logits_top_train_dump_10_top_1.json"

