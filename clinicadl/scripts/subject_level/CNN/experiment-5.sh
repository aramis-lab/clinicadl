#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=100:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=.
#SBATCH --output=outputs/scratch_baseline_job_%j.out
#SBATCH --error=outputs/scratch_baseline_job_%j.err
#SBATCH --job-name=3DADDL
#SBATCH --gres=gpu:1

export http_proxy=http://10.10.2.1:8123
export https_proxy=http://10.10.2.1:8123

module load pytorch/1.0.0_py27
# Network structure
SCRIPT="main.py"
MODEL="Conv5_FC3_mni"

# Dataset Management
COHORT='ADNI'
CAPS_EXT="_skull_stripping"
PREPROCESSING="mniskullstrip"
BASELINE=0
TASK='AD CN'
SPLITS=5
SPLIT=$1

# Training arguments
EPOCHS=50
BATCH=6
ACCUMULATION=2
EVALUATION=20
SAMPLER="random"
F_LR=1e-4
LR=1e-4
NORMALIZATION=1
PATIENCE=5
TOLERANCE=0

# Pretraining
T_BOOL=1
T_PATH="reproducibility_results/model-Conv5_FC3_preprocessing-mniskullstrip_task-autoencoder_baseline-1_norm-1_t-0_splits-5"

GPU=1
NUM_WORKERS=8
NUM_THREADS=0
OPTIONS=""
DATE="reproducibility_results"

TSVPATH="/network/lustre/iss01/home/elina.thibeausutre/data/Frontiers/$COHORT/lists_by_diagnosis/train/"
RESULTSPATH="/network/lustre/iss01/home/elina.thibeausutre/results/$DATE/"
IMGPATH="/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/$COHORT$CAPS_EXT"
T_PATH="/network/lustre/iss01/home/elina.thibeausutre/results/${T_PATH}"

TASK_NAME="${TASK// /_}"

if [ $BASELINE = 1 ]; then
echo "using only baseline data"
TASK_NAME="${TASK_NAME}_baseline"
OPTIONS="$OPTIONS --baseline"
PATIENCE=10
fi
echo $TASK_NAME

if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --gpu"
fi

if [ $NORMALIZATION = 1 ]; then
OPTIONS="${OPTIONS} -n"
fi

if [ $T_BOOL = 1 ]; then
OPTIONS="${OPTIONS} -t $T_PATH"
fi

NAME="model-${MODEL}_preprocessing-${PREPROCESSING}_task-${TASK_NAME}_norm-${NORMALIZATION}_t-${T_BOOL}"

if [ $SPLITS > 0 ]; then
echo "Use of $SPLITS-fold cross validation, split $SPLIT"
OPTIONS="$OPTIONS --n_splits $SPLITS --split $SPLIT"
NAME="${NAME}_splits-$SPLITS"
fi

python /network/lustre/iss01/home/elina.thibeausutre/AD-DL/Code/clinicadl/clinicadl/classifiers/three_d_cnn/subject_level/$SCRIPT $TSVPATH $RESULTSPATH$NAME $IMGPATH $MODEL -d $TASK -w $NUM_WORKERS --batch_size $BATCH --epochs $EPOCHS -asteps $ACCUMULATION -esteps $EVALUATION -lr $LR --num_threads $NUM_THREADS --sampler $SAMPLER --patience $PATIENCE --tolerance $TOLERANCE --preprocessing $PREPROCESSING $OPTIONS
