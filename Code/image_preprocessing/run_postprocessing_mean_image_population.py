####################

from T1_postprocessing_mean_img_population import get_mean_image_population

## run the pipeline
#  for test
##
caps_directory= '/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI'
tsv= '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_qc/ADNI_after_qc.tsv'

get_mean_image_population(caps_directory, tsv)
