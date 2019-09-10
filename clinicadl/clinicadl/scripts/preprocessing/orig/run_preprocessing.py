####################

from T1_preprocessing import preprocessing_t1w

## run the pipeline
#  for test
##
#bids_directory = '/teams/ARAMIS/PROJECTS/CLINICA/CLINICA_datasets/BIDS/ADNI_BIDS_T1_new'
#caps_directory= '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/CAPS'
#tsv= '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/image_preprocessing_test.tsv'
#working_dir = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Results/working_dir'
#ref_template = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/Data/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii'


bids_directory = '/network/lustre/dtlake01/aramis/projects/mauricio.diazmelo/tmp/DL/ADNI_BIDS_T1_new'
caps_directory= '/network/lustre/dtlake01/aramis/projects/mauricio.diazmelo/tmp/DL/ADNI_CAPS'
tsv= '/network/lustre/dtlake01/aramis/projects/mauricio.diazmelo/tmp/DL/image_preprocessing_test.tsv'
working_dir = '/localdrive10TB/data/mauricio.diazmelo/tmp/DL/working_dir'
ref_template = '/network/lustre/dtlake01/aramis/projects/mauricio.diazmelo/tmp/DL/mni_icbm152_nlin_sym_09c_nifti/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii'

wf = preprocessing_t1w(bids_directory, caps_directory, tsv, ref_template, working_directory=working_dir)
wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})
