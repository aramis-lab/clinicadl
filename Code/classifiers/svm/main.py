import argparse
from classification_utils import *
from model import *
from time import time

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
parser.add_argument("--diagnosis_tsv_folder", default='/teams/ARAMIS/PROJECTS/junhao.wen/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/tsv_after_data_splits/ADNI/lists_by_diagnosis/train',
                           help="Path to tsv file of the population based on the diagnosis tsv files. To note, the column name should be participant_id, session_id and diagnosis.")

parser.add_argument("--output_dir", default='/network/lustre/dtlake01/aramis/projects/clinica/CLINICA_datasets/CAPS/Frontiers_DL/Experiments_results/AD_CN/SVM_results/longitudinal',
                           help="Path to store the classification outputs, including log files for tensorboard usage and also the tsv files containg the performances.")
parser.add_argument('--baseline_or_longitudinal', default="longitudinal", choices=["baseline", "longitudinal"],
                    help="Using baseline scans or all available longitudinal scans for training")

parser.add_argument("--diagnoses_list", default=["AD", "CN"], type=str,
                    help="Labels for any binary task")

parser.add_argument("--n_splits", default=5, type=int,
                    help="Define the cross validation, by default, we use 5-fold.")
parser.add_argument("--group_id", default='ADNIbl', type=str,
                    help='The group ID of your image processing pipeline of Clinica SPM')

parser.add_argument("--num_workers", default=72, type=int,
                    help='the number of cpus to put to run the classification')

def main(options):

    ## to ensure have the same split with CNN approach
    splits_indices, diagnosis_tsv = extract_indices_from_5_fold(options.diagnosis_tsv_folder, options.n_splits, baseline_or_longitudinal=options.baseline_or_longitudinal, diagnoses_list=options.diagnoses_list)

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
