import pandas as pd
import argparse
from os import path
from copy import deepcopy, copy

from classification_utils import evaluate_prediction

parser = argparse.ArgumentParser(description="Argparser for evaluation of classifiers")

parser.add_argument("model_path", type=str,
                    help="Path to the trained model folder.")
parser.add_argument("cohort", type=str,
                    help="Name of the cohort.")
parser.add_argument("selection", type=str, choices=["loss", "acc"])


def majority_voting(results_df, subject, session):
    total_vote = 0
    print(subject, session)
    for i, result_df in enumerate(results_df):
        total_vote += result_df.loc[(subject, session), "predicted_label"]
        print("Vote %i: %i" % (i, result_df.loc[(subject, session), "predicted_label"]))

    print(total_vote)
    if total_vote < len(results_df) / 2:
        print("Vote 0")
        return 0
    print("Vote 1")
    return 1


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])

    # Loop on all folds trained
    performance_dir = path.join(options.model_path, 'performances')
    fold_directories = [path.join(performance_dir, 'fold_' + str(fold)) for fold in range(5)]
    results_df = [None] * 5

    for i, fold_dir in enumerate(fold_directories):
        results_df[i] = pd.read_csv(path.join(fold_dir, "best_" + options.selection, "test-" + options.cohort +
                                                     "_subject_level_result.tsv"), sep="\t")
        results_df[i].set_index(['participant_id', 'session_id'], inplace=True, drop=True)

    majority_df = copy(results_df[0])
    print(majority_df)

    for subject, session in majority_df.index.values:
        majority_df.loc[(subject, session), 'predicted_label'] = majority_voting(results_df, subject, session)

    majority_df.to_csv(path.join(performance_dir, 'best-' + options.selection + '_test-' + options.cohort +
                                 '_subject_level_results.tsv'), sep='\t')
    metrics = evaluate_prediction(majority_df.true_label, majority_df.predicted_label)
    del metrics['confusion_matrix']
    pd.DataFrame(metrics, index=[0]).to_csv(path.join(performance_dir, 'best-' + options.selection + '_test-' +
                                                      options.cohort + '_subject_level_metrics.tsv'),
                                            sep='\t')
