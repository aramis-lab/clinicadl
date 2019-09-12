#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --mem=17G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=.
#SBATCH --output=outputs/pytorch_job_%j.out
#SBATCH --error=outputs/pytorch_job_%j.err
#SBATCH --job-name=3DAE
#SBATCH --gres=gpu:1

export http_proxy=http://10.10.2.1:8123
export https_proxy=http://10.10.2.1:8123

module load pytorch/1.0.0_py27
# Network structure
SCRIPT="autoencoder_pretraining.py"
MODEL="Conv5_FC3"

# Dataset Management
COHORT='ADNI'
CAPS_EXT=""
PREPROCESSING='linear'
DIAGNOSES="AD CN MCI"
BASELINE=1
SPLITS=5
SPLIT=$1

# Training arguments
EPOCHS=50
BATCH=6
ACCUMULATION=2
EVALUATION=20
LR=1e-4
GREEDY_LEARNING=0
SIGMOID=0
NORMALIZATION=1
PATIENCE=50

# Pretraining
T_BOOL=0
T_PATH="/network/lustre/iss01/home/elina.thibeausutre/results/autoencoder_pretraining/model-Conv5_FC3_cohort-ADNI_task-autoencoder_gpu-1_workers-8_threads-0_epochs-100_lr-1e-4_norm-1_gl-0_sigmoid-0_batch-6_acc-2_eval-20_totalsplits-5/split-$SPLIT/model_best_loss.pth.tar"
T_DIFF=0

# Computation ressources
GPU=1
NUM_WORKERS=8
NUM_THREADS=0

OPTIONS=""
DATE="reproducibility_results"

TSVPATH="/network/lustre/iss01/home/elina.thibeausutre/data/Frontiers/$COHORT/lists_by_diagnosis/train"
RESULTSPATH="/network/lustre/iss01/home/elina.thibeausutre/results/$DATE/"
IMGPATH="/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/$COHORT$CAPS_EXT"

if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --gpu"
fi

if [ $GREEDY_LEARNING = 1 ]; then
OPTIONS="$OPTIONS --greedy_learning"
fi

if [ $SIGMOID = 1 ]; then
OPTIONS="${OPTIONS} --add_sigmoid"
fi

if [ $NORMALIZATION = 1 ]; then
OPTIONS="${OPTIONS} -n"
fi

if [ $T_BOOL = 1 ]; then
OPTIONS="$OPTIONS --pretrained_path $T_PATH -d $T_DIFF"
fi

if [ $BASELINE = 1 ]; then
echo "using only baseline data"
OPTIONS="$OPTIONS --baseline"
fi

NAME="model-${MODEL}_preprocessing-${PREPROCESSING}_task-autoencoder_baseline-${BASELINE}_norm-${NORMALIZATION}_t-${T_BOOL}"

if [ $SPLITS > 0 ]; then
echo "Use of $SPLITS-fold cross validation, split $SPLIT"
OPTIONS="$OPTIONS --n_splits $SPLITS --split $SPLIT"
NAME="${NAME}_splits-${SPLITS}"
fi

echo $NAME
echo $OPTIONS
python /network/lustre/iss01/home/elina.thibeausutre/AD-DL/Code/clinicadl/clinicadl/classifiers/three_d_cnn/subject_level/$SCRIPT \\
  $TSVPATH \\
  $RESULTSPATH$NAME \\
  $IMGPATH $MODEL \\
  -w $NUM_WORKERS \\
  --batch_size $BATCH \\
  --epochs $EPOCHS \\
  -asteps $ACCUMULATION \\
  -esteps $EVALUATION \\
  -lr $LR \\
  --num_threads $NUM_THREADS \\
  --diagnoses $DIAGNOSES \\
  --preprocessing $PREPROCESSING \\
  --patience $PATIENCE \\
  $OPTIONS --visualization
