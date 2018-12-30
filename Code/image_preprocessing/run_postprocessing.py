####################

from T1_postprocessing import postprocessing_t1w

## run the pipeline
#  for test
caps_directory = '/localdrive10TB/scratch/junhao.wen/Frontiers_DL/CAPS_ADNI'
tsv = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_qc/ADNI_after_qc.tsv'
patch_size = 21
stride_size = 21
working_dir = '/localdrive10TB/scratch/junhao.wen/Frontiers_DL/working_dir_ADNI_postprocessing'


wf = postprocessing_t1w(caps_directory, tsv, patch_size, stride_size, working_directory=working_dir)
wf.run(plugin='MultiProc', plugin_args={'n_procs': 72})
