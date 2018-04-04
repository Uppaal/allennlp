#!/bin/bash
#
#SBATCH --mem=30000
#SBATCH --job-name=1-gpu-bidaf-pytorch
#SBATCH --partition=m40-short
#SBATCH --output=ex_batch5_bidaf-pytorch-%A.out
#SBATCH --error=ex_batch5_bidaf-pytorch-%A.err
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

## Change this line so that it points to your bidaf github folder
cd /home/usaxena/work/s18/696/allennlp

# Training (Default - on SQuAD)
python -m allennlp.run train training_config/bidaf_extended_chars.json -s output_path_newsqa_batch5_extended_chars

# Evaluation (Default - on SQuAD)
# python -m allennlp.run evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz --evaluation-data-file https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json

# Evaluate on NewsQA
# python -m allennlp.run evaluate https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz --evaluation-data-file "data/newsqa_raw/test-v1.1.json"

# Prediction on a user defined passage
# echo '{"passage": "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.", "question": "How many partially reusable launch systems were developed?"}' > examples.jsonl
# python -m allennlp.run predict https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz  examples.jsonl

