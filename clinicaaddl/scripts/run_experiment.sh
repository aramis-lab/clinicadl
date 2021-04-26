#!/bin/bash
#SBATCH --gres=gpu:v100:2
#SBATCH --constraint="gpu"
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --mail-type=END
#SBATCH --mail-user=g.nasta.work@gmail.com
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/HLR_%j.out
#SBATCH -e logs/HLR_%j.err
echo $1
if [ -z "$2" ]
  then
    FROM_CHECKPOINT='False'
else
  FROM_CHECKPOINT=$2
fi
echo "Resume training from checkpoint: $FROM_CHECKPOINT"
module load anaconda/3/2020.02
module load cuda/10.2   
module load pytorch/gpu/1.6.0


BATCH=10
NUM_SPLITS=1
SPLIT=0
NPROC=2
# Other options
OPTIONS="--n_splits $NUM_SPLITS --split $SPLIT --nproc $NPROC --batch_size $BATCH"


# Computation ressources
GPU=1
if [ $GPU = 0 ]; then
OPTIONS="${OPTIONS} -cpu"
fi


# Run clinicaaddl
srun python3 $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py train \
  $1 --resume $FROM_CHECKPOINT $OPTIONS 
