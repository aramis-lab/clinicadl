####################

from quality_check_image_similarity import quality_check_image_similarity

## run the pipeline
#  for test
##
caps_directory= '/teams/ARAMIS/PROJECTS/CLINICA/CLINICA_datasets/temp/CAPS_ADNI_DL'
tsv= '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/ADNI_MCI_T1_rest.tsv'
working_dir = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/working_dir'
ref_template = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Data/mni_icbm152_nlin_sym_09c_nifti/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii'


wf = quality_check_image_similarity(caps_directory, tsv, ref_template, working_directory=working_dir)
wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})
