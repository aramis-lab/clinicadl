# coding: utf8
from Code.three_d_cnn.three_d_cnn import train_adni_mri

####################################
## RUN ADTI T1 CNN
####################################
### for T1
caps_directory = '/teams/ARAMIS/PROJECTS/simona.bottani/Multimodal_ADNI_M00/CAPS'
subjects_visits_tsv = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/machine_learning_classification/tsv_files/CN_vs_AD.tsv'
diagnoses_tsv = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/machine_learning_classification/tsv_files/CN_vs_AD_diagnosis.tsv'
n_fold = 5
batch_size = 10
num_epochs = 100
modality = 't1'
log_dir = '/teams/ARAMIS/PROJECTS/junhao.wen/DeepLearning/githubs/tensorflow_models/models/tutorials/3dcnn/adni_log/'
train_adni_mri(caps_directory, subjects_visits_tsv, diagnoses_tsv, n_fold, batch_size, num_epochs, log_dir, modality=modality)


# ## for dti
# caps_directory = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/machine_learning_classification/CAPS_ADNI_DTI/CAPS_final_ants_b_thomas'
# subjects_visits_tsv = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/machine_learning_classification/tsv_files/CN_vs_AD.tsv'
# diagnoses_tsv = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/machine_learning_classification/tsv_files/CN_vs_AD_diagnosis.tsv'
# n_fold = 5
# batch_size = 4
# num_epochs = 100
# modality = 'dti'
# log_dir = '/teams/ARAMIS/PROJECTS/junhao.wen/DeepLearning/githubs/tensorflow_models/models/tutorials/3dcnn/adni_log/'
# train_adni_mri(caps_directory, subjects_visits_tsv, diagnoses_tsv, n_fold, batch_size, num_epochs, log_dir, modality=modality)