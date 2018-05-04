#!/bin/bash
#
#SBATCH --mem=20000
#SBATCH --job-name=csv
#SBATCH --partition=titanx-short
#SBATCH --output=make-csv-%A.out
#SBATCH --error=make-csv-%A.err
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
cd /home/usaxena/work/s18/696/allen2/active_learning

# Logits 1
python process_logits.py --percent 1 --gpu 1 --source_logits_file "data/train-logits.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-logits-1.json" --target_dump_unselected_ids "slurm/not-needed.json" --score-type 0 --k 1

# Logits 10
#python process_logits.py --percent 10 --gpu 1 --source_logits_file "data/train-logits.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-logits-10.json" --target_dump_unselected_ids "slurm/not-needed.json" --score-type 0 --k 1

# Softmax 1
#python process_logits.py --percent 1 --gpu 1 --source_logits_file "data/train-logits.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-softmax-1.json" --target_dump_unselected_ids "slurm/not-needed.json" --score-type 1 --k 1

# Softmax 10
#python process_logits.py --percent 10 --gpu 1 --source_logits_file "data/train-logits.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-softmax-10.json" --target_dump_unselected_ids "slurm/not-needed.json" --score-type 1 --k 1

# Logits 1 top 1
#python process_logits.py --percent 1 --gpu 1 --source_logits_file "data/train-logits.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-logits-1-top1.json" --target_dump_unselected_ids "slurm/not-needed.json" --score-type 2 --k 1

# Logits 1 top 10
#python process_logits.py --percent 1 --gpu 1 --source_logits_file "data/logits-i1.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-logits-1-top10.json" --target_dump_unselected_ids "slurm/not-needed.json" --score-type 2 --k 10

# Softmax top 1
#python process_logits.py --percent 1 --gpu 1 --source_logits_file "data/logits-i1.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-softmax-top1.json" --target_dump_unselected_ids "slurm/meh.json" --score-type 3 --k 1

# Softmax top 10
#python process_logits.py --percent 1 --gpu 1 --source_logits_file "data/logits-i1.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-softmax-top10.json" --target_dump_unselected_ids "slurm/meh.json" --score-type 3 --k 10

# Logits 10 top 1
#python process_logits.py --percent 10 --gpu 1 --source_logits_file "data/train-logits.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-logits-10-top1.json" --target_dump_unselected_ids "slurm/not-needed.json" --score-type 2 --k 1

# Logits 10 top 10
#python process_logits.py --percent 10 --gpu 1 --source_logits_file "data/logits-i1.p" --source_file "/home/usaxena/work/s18/696/allen2/data/NewsQA/train-v1.1.json" --target_dump_selected_ids "slurm/i1-logits-10-top10.json" --target_dump_unselected_ids "slurm/not-needed.json" --score-type 2 --k 10

