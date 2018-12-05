import pandas as pd
import os

def check_dl_preprocessed(tsv, caps_dir):
	df = pd.read_csv(tsv, sep='\t')
	if 'participant_id' not in list(df.columns.values):
        	raise Exception('No participant_id column in TSV file.')
    	if 'session_id' not in list(df.columns.values):
        	raise Exception('No session_id column in TSV file.')
    	subjects = list(df.participant_id)
    	sessions = list(df.session_id)
	
	for i in xrange(len(subjects)):
		file = os.path.join(caps_dir, 'subjects', subjects[i], sessions[i], 't1', 'preprocessing_dl',  subjects[i] + '_' + sessions[i] + '_space-MNI_res-1x1x1_linear_registration.nii.gz')
		if os.path.isfile(file):
			pass
		else:
			print("!!! This image missing: %s" % file)



print("Begin:")
tsv = '/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_before_qc/qc_tsvs/ADNI_all_subjects.tsv'
caps_dir = '/teams/ARAMIS/PROJECTS/CLINICA/CLINICA_datasets/temp/CAPS_ADNI_DL'
check_dl_preprocessed(tsv, caps_dir)
