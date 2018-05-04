#!/bin/bash
#
#SBATCH --mem=50000
#SBATCH --job-name=yooo
#SBATCH --partition=m40-long
#SBATCH --output=yooo-%A.out
#SBATCH --error=yooo-%A.err
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

# When running this, change:
#  1. bidaf_finetune.json (directories only)
#  2. The lines below

# Change this line so that it points to your bidaf github folder
cd /home/usaxena/work/s18/696/allen2

# Depending on which experiment is being run, change this directory name
dir='logits-1'

# Iteration number
iprev='1'
i='2'

# Make directories
mkdir output_active/$dir
mkdir output_active/$dir/iteration_$iprev
mkdir data/active_learning/pipeline/$dir
mkdir data/active_learning/retained_target/$dir
mkdir data/active_learning/uncertain_logits/$dir
mkdir active_learning/data/$dir
mkdir active_learning/data/$dir/$i

# For first iteration only:
#cp output/squad/model.tar.gz output_active/$dir/iteration_0/model.tar.gz
#cp data/NewsQA/train-v1.1.json data/active_learning/retained_target/$dir/$iprev.json
#cp active_learning/data/train-logits.p active_learning/data/$dir/$i/logits.p
##cp data/active_learning/uncertain_logits/$dir.json data/active_learning/uncertain_logits/$dir/$i.json


# 1. Evaluate on unselected NewsQA (100-beta)%
python -m allennlp.run evaluate output_active/$dir/iteration_$iprev/model.tar.gz --evaluation-data-file "data/tiny.json" #"data/active_learning/retained_target/$dir/$iprev.json"

# 2. Select top k%
python active_learning/combine_logits.py --dir $dir --iteration $i
python active_learning/process_logits.py --gpu 1 --source_logits_file "active_learning/data/$dir/$i/logits.p" --source_file "data/active_learning/retained_target/$dir/$iprev.json" --target_dump_selected_ids "data/active_learning/uncertain_logits/$dir/$i.json" --target_dump_unselected_ids "data/active_learning/retained_target/$dir/$i.json" \
              --score-type 2 --k 10 --percent 1

# Add 100+1 code!!!!!!

# 3. Joint selected data with SQuAD
cp data/active_learning/uncertain_logits/$dir/$i.json data/active_learning/pipeline/$dir/train.json
python joint_train.py --target_sampling_ratio 0.0 --debug_ratio 1.0 --source_dir "data/squad/" --target_dir "data/active_learning/pipeline/$dir/" --output_dir "data/active_learning/pipeline/$dir/"
rm data/active_learning/pipeline/$dir/train.json

# 4. Finetune on joint
python -m allennlp.run fine-tune \
    --model-archive output_active/$dir/iteration_$iprev/model.tar.gz \
    --config-file training_config/bidaf-$dir-iteration$i.json \
    --serialization-dir output_active/$dir/iteration_$i

# 5. Evaluate on the test sets of SQuAD and NewsQA
python -m allennlp.run evaluate output_active/$dir/iteration_$i/model.tar.gz --evaluation-data-file "data/squad/dev-v1.1.json"
python -m allennlp.run evaluate output_active/$dir/iteration_$i/model.tar.gz --evaluation-data-file "data/NewsQA/test-v1.1.json"

# Remove additional files
rm -r "data/active_learning/pipeline/$dir/"
rm -r "active_learning/data/$dir/$i/eval_per_epoch/"
