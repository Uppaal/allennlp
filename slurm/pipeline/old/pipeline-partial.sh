#!/bin/bash
#
#SBATCH --mem=50000
#SBATCH --job-name=l-d10-t10
#SBATCH --partition=m40-long
#SBATCH --output=logits_top_train_dump_10_top_10-i1-%A.out
#SBATCH --error=logits_top_train_dump_10_top_10-i1-%A.err
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
#  3. The iteration number, depending on which iteration is being run. For example, when running the third iteration, change all iteration_0 to iteration_3, except for the last line of the script, which sould be iteration_4

# Change this line so that it points to your bidaf github folder
cd /home/usaxena/work/s18/696/allen2

# Depending on which experiment is being run, change this directory name
dir='logits_top_train_dump_10_top_10'

#------That's it! No need to change more. Sit back and relax.------#

# For first iteration, use pre-trained squad
mkdir output_active/$dir
mkdir output_active/$dir/iteration_0
mkdir data/active_learning/pipeline/$dir
cp output/squad/model.tar.gz output_active/$dir/iteration_0/model.tar.gz

# 1. Evaluate on NewsQA
#python -m allennlp.run evaluate output_active/$dir/iteration_0/model.tar.gz --evaluation-data-file "data/NewsQA/train-v1.1.json"

# 2. Select top k%
python active_learning/combine_logits.py --dir $dir --iteration 1
python active_learning/process_logits.py --percent 1 --gpu 1 --source_logits_file "active_learning/data/$dir/1/logits.p" --source_file "data/NewsQA/train-v1.1.json" --target_dump_selected_ids "data/active_learning/uncertain_logits/$dir/1.json" --target_dump_unselected_ids "data/active_learning/retained_target/$dir/1.json"

# 3. Join selected data with SQuAD
cp data/active_learning/uncertain_logits/$dir/1.json data/active_learning/pipeline/$dir/train.json
python joint_train.py --target_sampling_ratio 0.0 --debug_ratio 1.0 --source_dir "data/squad/" --target_dir "data/active_learning/pipeline/$dir/" --output_dir "data/active_learning/pipeline/$dir/"

# Now, train-v1.1.json and dev-v1.1.json are the files created, which need to be used. Delete train.json
rm data/active_learning/pipeline/$dir/train.json

# 4. Finetune on joint

# Use the model from the previous iteration (model-archive) and save the output in output_active (serialization-dir)
# Alter the bidaf_finetune file to change the number of epochs. And change the data directories to data/active_learning/pipeline/
python -m allennlp.run fine-tune \
    --model-archive output_active/$dir/iteration_0/model.tar.gz \
    --config-file training_config/bidaf_finetune.json \
    --serialization-dir output_active/$dir/iteration_1

# 5. Evaluate on the test sets of SQuAD and NewsQA
python -m allennlp.run evaluate output_active/$dir/iteration_1/model.tar.gz --evaluation-data-file "data/squad/dev-v1.1.json"
python -m allennlp.run evaluate output_active/$dir/iteration_1/model.tar.gz --evaluation-data-file "data/NewsQA/test-v1.1.json"

