def create_task_dict(task_list):
    """
    Creates a dictionnary of tasks
    - key is the name of the task (ex: 'AD_vs_CN')
    - associated content is the list of labels (ex: ['AD', 'CN'])

    :param task_list: task list given by the user
    :return: dict
    """

    task_dict = {}

    for task in task_list:
        list_diag = task.split('_')
        task_name = '_vs_'.join(list_diag)
        task_dict[task_name] = list_diag

    return task_dict


if __name__ == "__main__":
    import argparse
    import pandas as pd
    import os
    import shutil
    from os import path

    parser = argparse.ArgumentParser(description="Argparser for data formatting")

    # Mandatory arguments
    parser.add_argument("formatted_data_path", type=str,
                        help="Path to the folder containing formatted data.")

    # Modality selection
    parser.add_argument("--tasks", nargs="+", type=str,
                        default=None, help="Create lists with specific tasks. Labels must be separated by '_'. "
                                           "Ex: 'AD_CN'")

    args = parser.parse_args()

    # Create paths
    results_path = path.join(args.formatted_data_path, 'lists_by_diagnosis')
    train_path = path.join(results_path, 'train')
    test_path = path.join(results_path, 'test')

    # Task creation
    task_dict = create_task_dict(args.tasks)
    print(task_dict)
    if path.exists(path.join(args.formatted_data_path, 'lists_by_task')):
        shutil.rmtree(path.join(args.formatted_data_path, 'lists_by_task'))

    task_test_path = path.join(args.formatted_data_path, 'lists_by_task', 'test')
    if not path.exists(task_test_path):
        os.makedirs(task_test_path)

    if path.exists(train_path):
        task_train_path = path.join(args.formatted_data_path, 'lists_by_task', 'train')
        if not path.exists(task_train_path):
            os.makedirs(task_train_path)

    for task in task_dict.keys():
        task_test_df = pd.DataFrame()
        for diagnosis in task_dict[task]:
            diagnosis_df = pd.read_csv(path.join(test_path, diagnosis + '_baseline.tsv'), sep='\t')
            task_test_df = pd.concat([task_test_df, diagnosis_df])

        task_test_df.to_csv(path.join(task_test_path, task + '_baseline.tsv'), sep='\t', index=False)

        if path.exists(train_path):
            task_train_df = pd.DataFrame()
            task_train_complete_df = pd.DataFrame()
            for diagnosis in task_dict[task]:
                diagnosis_df = pd.read_csv(path.join(train_path, diagnosis + '_baseline.tsv'), sep='\t')
                task_train_df = pd.concat([task_train_df, diagnosis_df])
                diagnosis_df = pd.read_csv(path.join(train_path, diagnosis + '.tsv'), sep='\t')
                task_train_complete_df = pd.concat([task_train_complete_df, diagnosis_df])

            task_train_df.to_csv(path.join(task_train_path, task + '_baseline.tsv'), sep='\t', index=False)
            task_train_complete_df.to_csv(path.join(task_train_path, task + '.tsv'), sep='\t', index=False)
