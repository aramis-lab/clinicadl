import argparse
from classification_utils import *
from model import *

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"

parser = argparse.ArgumentParser(description="Argparser for SVM to apply the trained model on test datasets")

## data argument
parser.add_argument("--caps_directory_source", default='/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm',
                           help="Path to the caps Clinica SPM CAPS, where models was trained on")
parser.add_argument("--caps_directory_target", default='/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm',
                           help="Path to the caps Clinica SPM CAPS, where the test data were restored")
parser.add_argument("--diagnosis_tsv_test", default='/network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_task/test/AD_vs_CN_baseline.tsv',
                           help="Path to the tsv containing all the test dataset")

parser.add_argument("--output_dir_source", default='/home/junhao.wen/test',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument("--output_dir_target", default='/home/junhao.wen/test',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")

parser.add_argument('--n_splits', default=5,
                    help="how many folds")
parser.add_argument("--group_id_source", default='ADNIbl', type=str,
                    help='The group ID of your image processing pipeline of Clinica SPM')

parser.add_argument("--group_id_target", default='AIBLbl', type=str,
                    help='The group ID of your image processing pipeline of Clinica SPM')

parser.add_argument("--num_workers", default=8, type=int,
                    help='the number of cpus to put to run the classification')

def main(options):
    ## reconstruct the trained model from the source task
    diagnosis_tsv_all = os.path.join(options.output_dir_source, 'all_subjects.tsv')
    source_data = CAPSVoxelBasedInput(options.caps_directory_source, diagnosis_tsv_all, options.group_id_source,
                                      "T1", fwhm=8, mask_zeros=False)
    _, source_orig_shape, source_mask = load_data(source_data.get_images(), mask=True)

    target_data = CAPSVoxelBasedInput(options.caps_directory_target, options.diagnosis_tsv_test, options.group_id_target,
                                          "T1", fwhm=8, mask_zeros=False)

    result_dir = path.join(options.output_dir_target, 'test')
    if not path.exists(result_dir):
        os.makedirs(result_dir)
        subjects_visits = parsers.read_csv(diagnosis_tsv_target, sep='\t')
        if list(subjects_visits.columns.values) != ['participant_id', 'session_id', 'diagnosis']:
            raise Exception('Subjects and visits file is not in the correct format.')
    subjects_visits = parsers.read_csv(options.diagnosis_tsv_test, sep='\t')
    subjects = list(subjects_visits.participant_id)
    sessions = list(subjects_visits.session_id)
    subjects_sessions = [subjects[i] + '_' + sessions[i] for i in range(len(subjects))]


    for fi in range(options.n_splits):

        weights = np.loadtxt(path.join(options.output_dir_source, 'classifier', 'fold_' + str(fi), 'weights.txt'))

        w = revert_mask(weights, source_mask, source_orig_shape).flatten()
        b = np.loadtxt(path.join(options.output_dir_source, 'classifier', 'fold_' + str(fi), 'intersect.txt'))

        target_image = target_data.get_x()
        target_label = target_data.get_y()

        y_hat = np.dot(w, target_image.transpose()) + b
        y_binary = (y_hat < 0) * 1

        evaluation = evaluate_prediction(list(target_label), list(y_binary))

        del evaluation['confusion_matrix']
        if fi == 0:
            res_df = pd.DataFrame(evaluation, index=[fi])
            res_final = pd.DataFrame(columns=list(res_df.columns))
        else:
            res_df = pd.DataFrame(evaluation, index=[fi])
	# save the result for each subject in a tsv file 
        subjects_df = pd.DataFrame({'subject': np.array(subjects_sessions),
					'y': target_label,
                                        'y_hat': y_binary,
                                        'fold_index': fi})
        subjects_df.to_csv(path.join(result_dir, 'subjects_fold-' + str(fi) + '.tsv'),
                               index=False, sep='\t', encoding='utf-8')
        res_final = res_final.append(res_df)

#    result_dir = path.join(options.output_dir_target, 'test')
 #   if not path.exists(result_dir):
  #      os.makedirs(result_dir)
    res_final.to_csv(path.join(options.output_dir_target, 'test', 'results.tsv'), sep='\t', index=False)

if __name__ == "__main__":
    commandline = parser.parse_known_args()
    print("The commandline arguments:")
    print(commandline)
    ## save the commind line arguments into a tsv file for tracing all different kinds of experiments
    options = commandline[0]
    if commandline[1]:
        raise Exception("unknown arguments: %s" % (parser.parse_known_args()[1]))
    main(options)
