#!/bin/bash

#SBATCH --mem=30000
#SBATCH --job-name=al-bidaf
#SBATCH --output=dev1-al-%A.out
#SBATCH --error=dev1-al-%A.err


# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python/3.6.1


pip install --user nltk datefinder argparse pandas

## Change this line so that it points to your bidaf github folder
cd /home/usaxena/work/s18/696/allennlp/active_learning

python analyze.py -d "data/squad/train-v1.1.json" -t "data/squad_df.pkl"
python analyze.py -d "data/newsqa/train-dump.json" -t "data/newsqa_df.pkl"

