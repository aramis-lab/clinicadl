#!/bin/bash
#!/bin/bash
#SBATCH --gres=gpu:v100:2
#SBATCH --constraint="gpu"
#SBATCH --time=23:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=g.nasta.work@gmail.com
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/HLR_%j.out
#SBATCH -e logs//HLR_%j.err
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --workdir=/u/horlavanasta/MasterProject/Code/ClinicaTools/AD-DL/train/subject_level/cnn
#SBATCH --output=./exp1/pytorch_job_%A_%a.out
#SBATCH --error=./exp1/pytorch_job_%A_%a.err
#SBATCH --job-name=3CNN_subject
#SBATCH --gres=gpu:1

if [ -z "$1" ]
  then
    FROM_CHECKPOINT='False'
else
  FROM_CHECKPOINT=$2
fi
echo $FROM_CHECKPOINT
module load anaconda/3/2020.02
module load cuda/10.2   
module load pytorch/gpu/1.6.0

# Experiment training CNN

# Network structure
NETWORK="Conv5_FC3"
DATE="reproducibility_results_2"

# Input arguments to clinicaaddl
CAPS_DIR="/u/horlavanasta/MasterProject/ADNI_data/CAPSPreprocessedT1linear"
TSV_PATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/train"
OUTPUT_DIR="$SCRATCH/results/$DATE/"

# Computation ressources
NUM_PROCESSORS=8
GPU=1

# Dataset Management
PREPROCESSING='linear'
TASK='AD CN'
BASELINE=1
SPLITS=5
SPLIT=$SLURM_ARRAY_TASK_ID

# Training arguments
EPOCHS=50
BATCH=12
ACCUMULATION=2
EVALUATION=20
LR=1e-4
WEIGHT_DECAY=0
NORMALIZATION=0
PATIENCE=10
TOLERANCE=0

# Pretraining
T_BOOL=0
T_PATH=""
T_DIFF=0

# Other options
OPTIONS=""

if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --use_gpu"
fi

if [ $NORMALIZATION = 1 ]; then
OPTIONS="${OPTIONS} --minmaxnormalization"
fi

if [ $T_BOOL = 1 ]; then
OPTIONS="$OPTIONS --pretrained_path $T_PATH -d $T_DIFF"
fi
TASK_NAME="${TASK// /_}"

if [ $BASELINE = 1 ]; then
  echo "using only baseline data"
  TASK_NAME="${TASK_NAME}_baseline"
  OPTIONS="$OPTIONS --baseline"
fi
echo $TASK_NAME

NAME="subject_model-${NETWORK}_preprocessing-${PREPROCESSING}_task-${TASK_NAME}_norm-${NORMALIZATION}_t-${T_BOOL}"

if [ $SPLITS > 0 ]; then
echo "Use of $SPLITS-fold cross validation, split $SPLIT"
NAME="${NAME}_splits-${SPLITS}"
fi

echo $NAME

# Run clinicaaddl
clinicadl train \
  subject \
  $CAPS_DIR \
  $TSV_PATH \
  $OUTPUT_DIR$NAME \
  $NETWORK \
  --diagnoses $TASK \
  --nproc $NUM_PROCESSORS \
  --batch_size $BATCH \
  --evaluation_steps $EVALUATION \
  --preprocessing $PREPROCESSING \
  --n_splits $SPLITS \
  --split $SPLIT \
  --accumulation_steps $ACCUMULATION \
  --epochs $EPOCHS \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --patience $PATIENCE \
  $OPTIONS
