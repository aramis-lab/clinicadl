####################

from T1_postprocessing import postprocessing_t1w

## run the pipeline
#  for test
caps_directory = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/CAPS'
tsv = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/image_preprocessing_test.tsv'
patch_size = 21
stride_size = 21
working_dir = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/working_dir'


wf = postprocessing_t1w(caps_directory, tsv, patch_size, stride_size, working_directory=working_dir)
wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})
