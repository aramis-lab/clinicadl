####################

from T1_postprocessing_extract_hippo import postprocessing_t1w_extract_hippo

## run the pipeline
##
caps_directory= '/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/OASIS'
tsv='/network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/tsv_after_qc/OASIS_after_qc.tsv'
working_dir = '/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/working_dir/OASIS'

## for the left hippocampus
wf = postprocessing_t1w_extract_hippo(caps_directory, tsv, working_directory=working_dir, hemi='left')
wf.run(plugin='MultiProc', plugin_args={'n_procs': 28})

## for the right hippocampus
wf = postprocessing_t1w_extract_hippo(caps_directory, tsv, working_directory=working_dir, hemi='right')
wf.run(plugin='MultiProc', plugin_args={'n_procs': 28})
