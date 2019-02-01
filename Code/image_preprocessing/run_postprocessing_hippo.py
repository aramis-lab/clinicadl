####################

from T1_postprocessing_extract_hippo import postprocessing_t1w_extract_hippo

## run the pipeline
#  for test
##
caps_directory= '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/CAPS'
tsv= '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/image_preprocessing_test.tsv'
working_dir = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/working_dir'


## for the left hippocampus
wf = postprocessing_t1w_extract_hippo(caps_directory, tsv, working_directory=working_dir, hemi='left')
wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})

## for the right hippocampus
wf = postprocessing_t1w_extract_hippo(caps_directory, tsv, working_directory=working_dir, hemi='right')
wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})