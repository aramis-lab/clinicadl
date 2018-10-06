# coding: utf8
from Code.three_d_cnn.three_d_cnn import train_adni_mri

####################################
## RUN ADTI T1 CNN
####################################
### for T1
caps_directory = '/scratch/ARAMIS/users/junhao.wen/ADNI_DTI_CAPS/CAPS_final_ants_b_thomas'
subjects_visits_tsv = '/network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/CN_vs_AD.tsv'
diagnoses_tsv = '/network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/CN_vs_AD_diagnosis.tsv'
n_fold = 5
batch_size = 5
num_epochs = 70
#modality = 't1'
log_dir = '/network/lustre/iss01/home/junhao.wen/working_dir/3dcnn/adni_log/'


# ## for dti
modality = 'dti'
train_adni_mri(caps_directory, subjects_visits_tsv, diagnoses_tsv, n_fold, batch_size, num_epochs, log_dir, modality=modality)
