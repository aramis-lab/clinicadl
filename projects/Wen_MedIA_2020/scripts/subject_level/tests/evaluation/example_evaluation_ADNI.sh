#!/bin/bash
#SBATCH --partition=gpu_p1
#SBATCH --time=20:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1        # on réserve des coeurs physiques et non logiques
#SBATCH --ntasks=1
#SBATCH --workdir=/gpfswork/rech/zft/upd53tc/jobs2/AD-DL/tests/evaluation
#SBATCH --output=./subject_level/pytorch_job_%j.out
#SBATCH --error=./subject_level/pytorch_job_%j.err
#SBATCH --job-name=t_subject
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=mauricio.diaz@inria.fr

#export http_proxy=http://10.10.2.1:8123
#export https_proxy=http://10.10.2.1:8123

eval "$(conda shell.bash hook)"
conda activate clinicadl_env_py37

SCRIPT="evaluation_test.py"

# Data management
COHORT="OASIS_random"

BATCH=2
GPU=1
NUM_WORKERS=8
NUM_THREADS=0
OPTIONS=""
OUTPUTDIR="subject_model-Conv5_FC3_preprocessing-linear_task-AD_CN_norm-1_t-0_splits-5"

TSVPATH="$HOME/code/AD-DL/data/$COHORT/lists_by_diagnosis/test/"
RESULTSPATH="$SCRATCH/results/trivial_tests/3_test/$OUTPUTDIR/"
IMGPATH="$SCRATCH/../commun/datasets/${COHORT}"

if [ $GPU = 1 ]; then
OPTIONS="${OPTIONS} --gpu"
fi

python $HOME/code/AD-DL/clinicadl/clinicadl/subject_level/$SCRIPT \
  $RESULTSPATH \
  $IMGPATH \
  $TSVPATH \
  $COHORT \
  -w $NUM_WORKERS \
  --num_threads $NUM_THREADS $OPTIONS \
  --batch_size $BATCH
