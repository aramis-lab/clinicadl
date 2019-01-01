####################

from T1_postprocessing import postprocessing_t1w

## run the pipeline
#  for test
caps_directory = '/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI'
tsv = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_qc/ADNI_after_qc.tsv'
patch_size = 21
stride_size = 21
working_dir = '/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/run/junhao.wen'


wf = postprocessing_t1w(caps_directory, tsv, patch_size, stride_size, working_directory=working_dir, extract_method='slice')
wf.run(plugin='MultiProc', plugin_args={'n_procs': 72})
