####################

from T1_postprocessing_mean_img_population import get_mean_image_population

## run the pipeline
#  for test
##
caps_directory= '/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI'
tsv= '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_diagnosis/train/CN_baseline.tsv'
# tsv= '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/test.tsv'
template_image = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Data/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_cropped.nii.gz'


get_mean_image_population(caps_directory, tsv, template_image)
