#!/bin/bash
module load anaconda/3/2020.02
module load cuda/10.2
module load pytorch/gpu/1.6.0

# Experiment training CNN

# Importnant args
MS_MAIN='1.5T-3T'

    for NETWORK in "ResNet18" "SEResNet18" "ResNet18Expanded" "SEResNet18Expanded" "Conv5_FC3"
    do
        for LOSS in 'WeightedCrossEntropy' 'default'
        do
            for AUGMENTATION in True False
            do
                echo -e "==========================================================================================================\n"
                # Input arguments to clinicaaddl
                CAPS_DIR="$HOME/MasterProject/DataAndExperiments/Data/CAPS"
                TSV_PATH="$HOME/MasterProject/DataAndExperiments/Experiments/Experiments-${MS}/labels/test"
                OUTPUT_DIR="$HOME/MasterProject//DataAndExperiments/Experiments/Experiments-${MS}/NNs/${NETWORK}/"

                # Dataset Management
                PREPROCESSING='linear'
                TASK='AD CN'
                BASELINE=True
                TASK_NAME="${TASK// /_}"
                POSTFIX="test_${MS}"


            NAME="subject_model-${NETWORK}_preprocessing-${PREPROCESSING}_task-${TASK_NAME}_norm-${NORMALIZATION}_loss-${LOSS}_augm${AUGMENTATION}"
                # echo $NAME

                # Run clinicaaddl
                python $HOME/MasterProject/Code/ClinicaTools/AD-DL/clinicaaddl/clinicaaddl/main.py classify \
                $CAPS_DIR \
                $TSV_PATH \
                $OUTPUT_DIR$NAME \
                $POSTFIX \
                --selection_metrics balanced_accuracy loss last_checkpoint

            done
        done
    done

