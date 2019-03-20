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

parser = argparse.ArgumentParser(description="Argparser for SVM")

## data argument
parser.add_argument("--caps_directory", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/ADNI_clinica_spm',
                           help="Path to the caps Clinica SPM CAPS")
parser.add_argument("--diagnosis_tsv_folder", default='/network/lustre/iss01/home/junhao.wen/Project/AD-DL/tsv_files/tsv_after_data_splits/Elina_version/data/ADNI/lists_by_diagnosis/train',
                           help="Path to tsv file of the population based on the diagnosis tsv files. To note, the column name should be participant_id, session_id and diagnosis.")

parser.add_argument("--output_dir", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/sMCI_pMCI/SVM_results/transfer_learning/longitudinal',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument('--baseline_or_longitudinal', default="longitudinal", choices=["baseline", "longitudinal"],
                    help="Using baseline scans or all available longitudinal scans for training")

parser.add_argument("--diagnoses_list", default=["sMCI", "pMCI"], type=str,
                    help="Labels for any binary task")
parser.add_argument("--transfer_learning", default=True, type=bool,
                    help="If do transfer learning corss task, for SVM, it will not retrain the model because its loss if convex, it will apply the model directly")
parser.add_argument("--source_classifer_dir", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/longitudinal',
                           help="Path to store the classifier of the source task, e.g. AD vs CN.")


parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--group_id", default='ADNIbl', type=str,
                    help='The group ID of your image processing pipeline of Clinica SPM')

parser.add_argument("--num_workers", default=8, type=int,
                    help='the number of cpus to put to run the classification')

def main(options):

    if options.transfer_learning:
        print "Do transfer learning fro SVM:"

        ## source tsv
        splits_indices_source, diagnosis_tsv_source = extract_indices_from_5_fold(options.diagnosis_tsv_folder, options.n_splits,
                                                                    baseline_or_longitudinal=options.baseline_or_longitudinal,
                                                                    diagnoses_list=["AD", "CN"])
        ## target tsv
        splits_indices_target, diagnosis_tsv_target = extract_indices_from_5_fold(options.diagnosis_tsv_folder, options.n_splits,
                                                                    baseline_or_longitudinal=options.baseline_or_longitudinal,
                                                                    diagnoses_list=options.diagnoses_list)

        ## reconstruct the trained model from the source task
        source_data = CAPSVoxelBasedInput(options.caps_directory, diagnosis_tsv_source, options.group_id,
                                                "T1", fwhm=8, mask_zeros=False)
        target_data = CAPSVoxelBasedInput(options.caps_directory, diagnosis_tsv_target, options.group_id,
                                                "T1", fwhm=8, mask_zeros=False)

        source_x, source_orig_shape, source_mask = load_data(source_data.get_images(), mask=True)

        weights = np.loadtxt(path.join(options.source_classifer_dir, 'classifier', 'weights.txt'))
        w = revert_mask(weights, source_mask, source_orig_shape).flatten()
        b = np.loadtxt(path.join(options.source_classifer_dir, 'classifier', 'intersect.txt'))

        ##Only apply the pre-trained model from AD vs CN to the validation data
        for fi in range(options.n_splits):
            target_image = target_data.get_x()[list(splits_indices_target[fi][1])]
            target_label = target_data.get_y()[list(splits_indices_target[fi][1])]

            y_hat = np.dot(w, target_image.transpose()) + b
            y_binary = (y_hat < 0) * 1.0

            evaluation = evaluate_prediction(target_label, y_binary)

            del evaluation['confusion_matrix']
            if fi == 0:
                res_df = pd.DataFrame(evaluation, index=[fi])
                res_final = pd.DataFrame(columns=list(res_df.columns))
            else:
                res_df = pd.DataFrame(evaluation, index=[fi])

            res_final = res_final.append(res_df)

        res_final.to_csv(path.join(options.output_dir, 'results.tsv'), sep='\t', index=False)

    else:
        print "Training from scratch:"
        ## to ensure have the same split with CNN approach
        splits_indices, diagnosis_tsv = extract_indices_from_5_fold(options.diagnosis_tsv_folder, options.n_splits,
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
