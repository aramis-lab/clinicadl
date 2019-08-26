import argparse
from classification_utils import *
from model import *
import numpy as np

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for SVM")

## data argument
parser.add_argument("--caps_directory", default='/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm',
                           help="Path to the caps Clinica SPM CAPS")
parser.add_argument("--diagnosis_tsv_folder", default='/network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_diagnosis/test_to_understand/',
                           help="Path to tsv file of the population based on the diagnosis tsv files. To note, the column name should be participant_id, session_id and diagnosis.")

# parser.add_argument("--output_dir", default='/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/sMCI_pMCI/SVM_results/transfer_learning/longitudinal',
#                            help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")

parser.add_argument("--output_dir", default='/home/junhao.wen/test',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")

parser.add_argument('--baseline_or_longitudinal', default="baseline", choices=["baseline", "longitudinal"],
                    help="Using baseline scans or all available longitudinal scans for training")

parser.add_argument("--diagnoses_list", default=["pMCI", "sMCI"], type=str,
                    help="Labels for any binary task")
parser.add_argument("--transfer_learning", default=True, type=bool,
                    help="If do transfer learning corss task, for SVM, it will not retrain the model because its loss if convex, it will apply the model directly")
parser.add_argument("--source_classifer_dir", default='/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/baseline',
                           help="Path to store the classifier of the source task, e.g. AD vs CN.")


parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--group_id", default='ADNIbl', type=str,
                    help='The group ID of your image processing pipeline of Clinica SPM')

parser.add_argument("--num_workers", default=8, type=int,
                    help='the number of cpus to put to run the classification')

def main(options):

    if options.transfer_learning:
        print "Do transfer learning for SVM:"
        ## source tsv
        diagnosis_tsv_all = os.path.join(options.source_classifer_dir, 'all_subjects.tsv')
        ## reconstruct the trained model from the source task
        source_data = CAPSVoxelBasedInput(options.caps_directory, diagnosis_tsv_all, options.group_id,
                                          "T1", fwhm=8, mask_zeros=False)
        _, source_orig_shape, source_mask = load_data(source_data.get_images(), mask=True)

	splits_indices_target, diagnosis_tsv_target = extract_indices_from_5_fold(options.diagnosis_tsv_folder, options.n_splits, options.output_dir,
                                                                        baseline_or_longitudinal="baseline",
                                                                        diagnoses_list=options.diagnoses_list)

        target_data = CAPSVoxelBasedInput(options.caps_directory, diagnosis_tsv_target, options.group_id,
                                                    "T1", fwhm=8, mask_zeros=False)	
        subjects_visits = parsers.read_csv(diagnosis_tsv_target, sep='\t')
        if list(subjects_visits.columns.values) != ['participant_id', 'session_id', 'diagnosis']:
            raise Exception('Subjects and visits file is not in the correct format.')
        subjects = list(subjects_visits.participant_id)
        sessions = list(subjects_visits.session_id)
	subjects_sessions = [subjects[i] + '_' + sessions[i] for i in range(len(subjects))]

        for fi in range(options.n_splits):
            weights = np.loadtxt(path.join(options.source_classifer_dir, 'classifier', 'fold_' + str(fi), 'weights.txt'))
            w = revert_mask(weights, source_mask, source_orig_shape).flatten()
            b = np.loadtxt(path.join(options.source_classifer_dir, 'classifier', 'fold_' + str(fi), 'intersect.txt'))

            ##Only apply the pre-trained model from AD vs CN to the validation data
            target_image = target_data.get_x()[list(splits_indices_target[fi][1])]
            target_label = target_data.get_y()[list(splits_indices_target[fi][1])]

            y_hat = np.dot(w, target_image.transpose()) + b
            y_binary = (y_hat < 0) * 1

            evaluation = evaluate_prediction(target_label, y_binary)

            del evaluation['confusion_matrix']
            if fi == 0:
                res_df = pd.DataFrame(evaluation, index=[fi])
                res_final = pd.DataFrame(columns=list(res_df.columns))
            else:
                res_df = pd.DataFrame(evaluation, index=[fi])
            subjects_df = pd.DataFrame({'subject': np.array(subjects_sessions)[list(splits_indices_target[fi][1])],
					'y': target_label,
                                        'y_hat': y_binary,
                                        'y_index': splits_indices_target[fi][1]})
	    subjects_df.to_csv(path.join(options.output_dir, 'subjects_fold-' + str(fi) + '.tsv'), index=False, sep='\t', encoding='utf-8')
            res_final = res_final.append(res_df)

            res_final.to_csv(path.join(options.output_dir, 'results.tsv'), sep='\t', index=False)

    else:
        print "Training from scratch:"
        ## to ensure have the same split with CNN approach
        splits_indices, diagnosis_tsv = extract_indices_from_5_fold(options.diagnosis_tsv_folder, options.n_splits, options.output_dir,
                                                                    baseline_or_longitudinal=options.baseline_or_longitudinal,
                                                                    diagnoses_list=options.diagnoses_list)

        wf = VB_KFold_DualSVM(options.caps_directory, diagnosis_tsv, options.group_id, "T1", options.output_dir,
                              n_threads=options.num_workers, n_folds=options.n_splits, splits_indices=splits_indices)

        wf.run()

if __name__ == "__main__":
    commandline = parser.parse_known_args()
    print("The commandline arguments:")
    print(commandline)
    ## save the commind line arguments into a tsv file for tracing all different kinds of experiments
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
