#!/bin/bash
#
#SBATCH --mem=50000
#SBATCH --job-name=j10
#SBATCH --partition=m40-long
#SBATCH --output=joint-10-i1-%A.out
#SBATCH --error=joint-10-i1-%A.err
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


# Change this line so that it points to your bidaf github folder
cd /home/usaxena/work/s18/696/allen2

# Depending on which experiment is being run, change this directory name
dir='joint-10'

# Iteration number
iprev='0'
i='1'

# Make directories
#mkdir data/total
#mkdir data/total/$dir
mkdir output_joint
mkdir output_joint/$dir
#mkdir data/active_learning/pipeline/$dir
#mkdir data/active_learning/retained_target/$dir
#mkdir data/active_learning/uncertain_logits/$dir
#mkdir active_learning/data/$dir
#mkdir active_learning/data/$dir/$i

# For first iteration only:
cp output/squad/model.tar.gz output_joint/$dir/model.tar.gz
#cp data/NewsQA/train-v1.1.json data/active_learning/retained_target/$dir/$iprev.json
#cp active_learning/data/train-logits.p active_learning/data/$dir/$i/logits.p
###cp data/active_learning/uncertain_logits/$dir/$iprev.json
##cp data/active_learning/uncertain_logits/$dir.json data/active_learning/uncertain_logits/$dir/$i.json



mkdir data/vanilla_joint
mkdir data/vanilla_joint/$dir

# Join selected data with (SQuAD + NQA)
python joint_train_old.py --target_sampling_ratio 0.0 --debug_ratio 0.01 --source_dir "data/squad/" --target_dir "data/NewsQA/" --output_dir "data/vanilla_joint/$dir/"

echo "------------Datsets created. Starting fine-tuning-----------------"

# 4. Finetune on joint
python -m allennlp.run fine-tune \
    --model-archive output_joint/$dir/model.tar.gz \
    --config-file training_config/$dir.json \
    --serialization-dir output_joint/$dir/iteration_$i

echo "-----------Training complete. Evaluating on test sets-------------"

# 5. Evaluate on the test sets of SQuAD and NewsQA
python -m allennlp.run evaluate output_joint/$dir/iteration_$i/model.tar.gz --evaluation-data-file "data/squad/dev-v1.1.json"
python -m allennlp.run evaluate output_joint/$dir/iteration_$i/model.tar.gz --evaluation-data-file "data/NewsQA/test-v1.1.json"
