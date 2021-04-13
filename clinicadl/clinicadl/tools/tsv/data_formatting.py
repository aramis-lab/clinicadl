# coding: utf8

"""
Source files can be obtained by running the following commands on a BIDS folder:
 - clinica iotools merge-tsv
 - clinica iotools missing-mods
To download Clinica follow the instructions at http://www.clinica.run/doc/#installation

NB: Other preprocessing may be needed on the merged file obtained: for example the selection of subjects older than 62
in the OASIS dataset is not done in this script. Moreover a quality check may be needed at the end of preprocessing
pipelines, leading to the removal of some subjects.
"""
from ..deep_learning.iotools import return_logger, commandline_to_json
from .tsv_utils import neighbour_session, last_session, after_end_screening, find_label, first_session
import pandas as pd
from os import path
from copy import copy
import numpy as np
import logging
import os


def cleaning_nan_diagnoses(bids_df, logger):
    """
    Printing the number of missing diagnoses and filling it partially for ADNI datasets

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        logger: Logger object from logging library

    Returns:
        cleaned DataFrame
    """
    bids_copy_df = copy(bids_df)

    # Look for the diagnosis in another column in ADNI
    if 'adni_diagnosis_change' in bids_df.columns:
        change_dict = {1: 'CN', 2: 'MCI', 3: 'AD', 4: 'MCI', 5: 'AD', 6: 'AD', 7: 'CN', 8: 'MCI', 9: 'CN', -1: np.nan}

        missing_diag = 0
        found_diag = 0

        for subject, session in bids_df.index.values:
            diagnosis = bids_df.loc[(subject, session), 'diagnosis']
            if isinstance(diagnosis, float):
                missing_diag += 1
                change = bids_df.loc[(subject, session), 'adni_diagnosis_change']
                if not np.isnan(change) and change != -1:
                    found_diag += 1
                    bids_copy_df.loc[(subject, session), 'diagnosis'] = change_dict[change]

    else:
        missing_diag = 0
        found_diag = 0

        for subject, session in bids_df.index.values:
            diagnosis = bids_df.loc[(subject, session), 'diagnosis']
            if isinstance(diagnosis, float):
                missing_diag += 1

    logger.debug('Missing diagnoses: %i' % missing_diag)
    logger.debug('Missing diagnoses not found: %i' % (missing_diag - found_diag))

    return bids_copy_df


def infer_or_drop_diagnosis(bids_df, logger):
    """
    Deduce the diagnosis when missing from previous and following sessions of the subject. If not identical, the session
    is dropped. Sessions with no diagnosis are also dropped when there are the last sessions of the follow-up.

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        logger: Logger object from logging library

    Returns:
        cleaned DataFrame
    """
    bids_copy_df = copy(bids_df)
    found_diag_interpol = 0

    for subject, subject_df in bids_df.groupby(level=0):
        session_list = [int(session[5::]) for _, session in subject_df.index.values]

        for _, session in subject_df.index.values:
            diagnosis = subject_df.loc[(subject, session), 'diagnosis']
            session_nb = int(session[5::])

            if isinstance(diagnosis, float):
                if session == last_session(session_list):
                    bids_copy_df.drop((subject, session), inplace=True)
                else:
                    prev_session = neighbour_session(session_nb, session_list, -1)
                    prev_diagnosis = bids_df.loc[(subject, prev_session), 'diagnosis']
                    while isinstance(prev_diagnosis, float) and prev_session != first_session(subject_df):
                        prev_session = neighbour_session(int(prev_session[5::]), session_list, -1)
                        prev_diagnosis = bids_df.loc[(subject, prev_session), 'diagnosis']
                    post_session = neighbour_session(session_nb, session_list, +1)
                    post_diagnosis = bids_df.loc[(subject, post_session), 'diagnosis']
                    while isinstance(post_diagnosis, float) and post_session != last_session(session_list):
                        post_session = neighbour_session(int(post_session[5::]), session_list, +1)
                        post_diagnosis = bids_df.loc[(subject, post_session), 'diagnosis']
                    if prev_diagnosis == post_diagnosis:
                        found_diag_interpol += 1
                        bids_copy_df.loc[(subject, session), 'diagnosis'] = prev_diagnosis
                    else:
                        bids_copy_df.drop((subject, session), inplace=True)

    logger.debug('Inferred diagnosis: %i' % found_diag_interpol)

    return bids_copy_df


def mod_selection(bids_df, missing_mods_dict, mod='t1w'):
    """
    Select only sessions for which the modality is present

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        missing_mods_dict: dictionnary of the DataFrames of missing modalities
        mod: the modality used for selection

    Returns:
        DataFrame
    """
    bids_copy_df = copy(bids_df)
    if mod is not None:
        for subject, session in bids_df.index.values:
            try:
                mod_present = missing_mods_dict[session].loc[subject, mod]
                if not mod_present:
                    bids_copy_df.drop((subject, session), inplace=True)
            except Exception:
                bids_copy_df.drop((subject, session), inplace=True)

    return bids_copy_df


def stable_selection(bids_df, diagnosis='AD', logger=None):
    """
    Select only subjects whom diagnosis is identical during the whole follow-up.

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        logger: Logger object from logging library
        diagnosis: (str) diagnosis selected

    Returns:
        DataFrame containing only the patients a the stable diagnosis
    """
    if logger is None:
        logger = logging
        logger.basicConfig(level=logging.DEBUG)

    # Keep diagnosis at baseline
    bids_df = bids_df[bids_df.baseline_diagnosis == diagnosis]
    bids_df = cleaning_nan_diagnoses(bids_df, logger=logger)

    # Drop if not stable
    bids_copy_df = copy(bids_df)
    n_subjects = 0
    for subject, subject_df in bids_df.groupby(level=0):
        subject_drop = False
        try:
            diagnosis_bl = subject_df.loc[(subject, 'ses-M00'), 'baseline_diagnosis']
        except KeyError:
            raise KeyError("The baseline session is necessary for labels selection. It is missing for subject %s"
                           % subject)
        diagnosis_values = subject_df.diagnosis.values
        for diagnosis in diagnosis_values:
            if not isinstance(diagnosis, float):
                if diagnosis != diagnosis_bl:
                    subject_drop = True
                    n_subjects += 1

        if subject_drop:
            bids_copy_df.drop(subject, inplace=True)
    bids_df = copy(bids_copy_df)
    logger.debug('Number of unstable subjects dropped: %i' % n_subjects)

    bids_df = infer_or_drop_diagnosis(bids_df, logger)
    return bids_df


def mci_stability(bids_df, horizon_time=36, logger=None):
    """
    A method to label all MCI sessions depending on their stability on the time horizon

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        horizon_time: (int) time horizon in months
        logger: Logger object from logging library

    Returns:
        DataFrame with new labels
    """
    if logger is None:
        logger = logging
        logger.basicConfig(level=logging.DEBUG)

    diagnosis_list = ['MCI', 'EMCI', 'LMCI']
    bids_df = bids_df[(bids_df.baseline_diagnosis.isin(diagnosis_list))]
    bids_df = cleaning_nan_diagnoses(bids_df, logger)
    bids_df = infer_or_drop_diagnosis(bids_df, logger)

    # Check possible double change in diagnosis in time
    bids_copy_df = copy(bids_df)
    nb_subjects = 0
    for subject, subject_df in bids_df.groupby(level=0):
        session_list = [int(session[5::]) for _, session in subject_df.index.values]
        session_list.sort()
        diagnosis_list = []
        for session in session_list:
            if session < 10:
                diagnosis_list.append(bids_df.loc[(subject, 'ses-M0' + str(session)), 'diagnosis'])
            else:
                diagnosis_list.append(bids_df.loc[(subject, 'ses-M' + str(session)), 'diagnosis'])

        new_diagnosis = diagnosis_list[0]
        nb_change = 0
        for diagnosis in diagnosis_list:
            if new_diagnosis != diagnosis:
                new_diagnosis = diagnosis
                nb_change += 1

        if nb_change > 1:
            nb_subjects += 1
            bids_copy_df.drop(subject, inplace=True)

    logger.debug('Dropped subjects: %i' % nb_subjects)
    bids_df = copy(bids_copy_df)

    # Stability of sessions
    stability_dict = {'CN': 'r', 'MCI': 's', 'AD': 'p'}  # Do not take into account the case of missing diag = nan

    bids_copy_df = copy(bids_df)
    for subject, subject_df in bids_df.groupby(level=0):
        session_list = [int(session[5::]) for _, session in subject_df.index.values]
        # print(subject_df.diagnosis)
        for _, session in subject_df.index.values:
            diagnosis = subject_df.loc[(subject, session), 'diagnosis']

            # If the diagnosis is not MCI we remove the time point
            if diagnosis != 'MCI':
                bids_copy_df.drop((subject, session), inplace=True)

            else:
                session_nb = int(session[5::])
                horizon_session_nb = session_nb + horizon_time
                horizon_session = 'ses-M' + str(horizon_session_nb)
                # print(session, '-->', horizon_session)

                if horizon_session_nb in session_list:
                    horizon_diagnosis = subject_df.loc[(subject, horizon_session), 'diagnosis']
                    update_diagnosis = stability_dict[horizon_diagnosis] + 'MCI'
                    # print(horizon_diagnosis, update_diagnosis)
                    bids_copy_df.loc[(subject, session), 'diagnosis'] = update_diagnosis
                else:
                    if after_end_screening(horizon_session_nb, session_list):
                        # Two situations, change in last session AD or CN --> pMCI or rMCI
                        # Last session MCI --> uMCI
                        last_diagnosis = subject_df.loc[(subject, last_session(session_list)), 'diagnosis']
                        # This section must be discussed --> removed in Jorge's paper
                        if last_diagnosis != 'MCI':
                            update_diagnosis = stability_dict[last_diagnosis] + 'MCI'
                        else:
                            update_diagnosis = 'uMCI'
                        # print(update_diagnosis)
                        bids_copy_df.loc[(subject, session), 'diagnosis'] = update_diagnosis

                    else:
                        prev_session = neighbour_session(horizon_session_nb, session_list, -1)
                        post_session = neighbour_session(horizon_session_nb, session_list, +1)
                        # print('prev_session', prev_session)
                        # print('post_session', post_session)
                        prev_diagnosis = subject_df.loc[(subject, prev_session), 'diagnosis']
                        if prev_diagnosis != 'MCI':
                            update_diagnosis = stability_dict[prev_diagnosis] + 'MCI'
                        else:
                            post_diagnosis = subject_df.loc[(subject, post_session), 'diagnosis']
                            if post_diagnosis != 'MCI':
                                update_diagnosis = 'uMCI'
                            else:
                                update_diagnosis = 'sMCI'
                        # print(update_diagnosis)
                        bids_copy_df.loc[(subject, session), 'diagnosis'] = update_diagnosis

    return bids_copy_df


def diagnosis_removal(MCI_df, diagnosis_list):
    """
    Removes subjects whom last diagnosis is in the list provided (avoid to keep rMCI and pMCI in sMCI lists).

    Args:
        MCI_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        diagnosis_list: list of diagnoses that will be removed

    Returns:
        cleaned DataFrame
    """

    output_df = copy(MCI_df)

    # Remove subjects who regress to CN label, even late in the follow-up
    for subject, subject_df in MCI_df.groupby(level=0):
        session_list = [int(session[5::]) for _, session in subject_df.index.values]
        last_session_id = last_session(session_list)
        last_diagnosis = subject_df.loc[(subject, last_session_id), 'diagnosis']
        if last_diagnosis in diagnosis_list:
            output_df.drop(subject, inplace=True)

    return output_df


def apply_restriction(bids_df, restriction_path):
    """
    Application of a restriction (for example after the removal of some subjects after a preprocessing pipeline)

    Args:
        bids_df: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis']
        restriction_path: DataFrame with columns including ['participant_id', 'session_id', 'diagnosis'] including
            all the sessions that can be included

    Returns:
        The restricted DataFrame
    """
    bids_copy_df = copy(bids_df)

    if restriction_path is not None:
        restriction_df = pd.read_csv(restriction_path, sep='\t')

        for subject, session in bids_df.index.values:
            subject_qc_df = restriction_df[(restriction_df.participant_id == subject) & (restriction_df.session_id == session)]
            if len(subject_qc_df) != 1:
                bids_copy_df.drop((subject, session), inplace=True)

    return bids_copy_df


def get_labels(merged_tsv, missing_mods, results_path,
               diagnoses, modality="t1w", restriction_path=None,
               time_horizon=36, variables_of_interest=None,
               remove_smc=True, verbose=0):
    """
    Writes one tsv file per label in diagnoses argument based on merged_tsv and missing_mods.

    Args:
        merged_tsv (str): Path to the file obtained by the command clinica iotools merge-tsv.
        missing_mods (str): Path to the folder where the outputs of clinica iotools missing-mods are.
        results_path (str): Path to the folder where tsv files are extracted.
        diagnoses (list): Labels that must be extracted from merged_tsv.
        modality (str): Modality to select sessions. Sessions which do not include the modality will be excluded.
        restriction_path (str): Path to a tsv containing the sessions that can be included.
        time_horizon (int): Time horizon to analyse stability of MCI subjects.
        variables_of_interest (list[str]): columns that should be kept in the output tsv files.
        remove_smc (bool): if True SMC participants are removed from the lists.
        verbose (int): level of verbosity.

    Returns:
         writes one tsv file per label at results_path/<label>.tsv
    """
    logger = return_logger(verbose, "getlabels")

    commandline_to_json({
        "output_dir": results_path,
        "merged_tsv": merged_tsv,
        "missing_mods": missing_mods,
        "diagnoses": diagnoses,
        "modality": modality,
        "restriction_path": restriction_path,
        "time_horizon": time_horizon,
        "variables_of_interest": variables_of_interest,
        "remove_smc": remove_smc
    }, filename="getlabels.json")

    # Reading files
    bids_df = pd.read_csv(merged_tsv, sep='\t')
    bids_df.set_index(['participant_id', 'session_id'], inplace=True)
    variables_list = ["diagnosis"]
    try:
        variables_list.append(find_label(bids_df.columns.values, 'age'))
        variables_list.append(find_label(bids_df.columns.values, 'sex'))
    except ValueError:
        logger.warn("The age or sex values were not found in the dataset.")
    if variables_of_interest is not None:
        variables_set = set(variables_of_interest) | set(variables_list)
        variables_list = list(variables_set)

    list_files = os.listdir(missing_mods)
    missing_mods_dict = {}

    for file in list_files:
        filename, fileext = path.splitext(file)
        if fileext == '.tsv':
            session = filename.split('_')[-1]
            missing_mods_df = pd.read_csv(path.join(missing_mods, file), sep='\t')
            if len(missing_mods_df) == 0:
                raise ValueError("Empty DataFrame at path %s" % path.join(missing_mods, file))

            missing_mods_df.set_index('participant_id', drop=True, inplace=True)
            missing_mods_dict[session] = missing_mods_df

    # Creating results path
    os.makedirs(results_path, exist_ok=True)

    # Remove SMC patients
    if remove_smc:
        if "diagnosis_bl" in bids_df.columns.values:  # Retro-compatibility
            bids_df = bids_df[~(bids_df.diagnosis_bl == "SMC")]
        if "diagnosis_sc" in bids_df.columns.values:
            bids_df = bids_df[~(bids_df.diagnosis_sc == "SMC")]

    # Adding the field baseline_diagnosis
    bids_copy_df = copy(bids_df)
    bids_copy_df['baseline_diagnosis'] = pd.Series(np.zeros(len(bids_df)), index=bids_df.index)
    for subject, subject_df in bids_df.groupby(level=0):
        baseline_diagnosis = subject_df.loc[(subject, 'ses-M00'), 'diagnosis']
        bids_copy_df.loc[subject, 'baseline_diagnosis'] = baseline_diagnosis

    bids_df = copy(bids_copy_df)

    time_MCI_df = None
    if 'AD' in diagnoses:
        logger.info('Beginning the selection of AD label')
        output_df = stable_selection(bids_df, diagnosis='AD', logger=logger)
        output_df = mod_selection(output_df, missing_mods_dict, modality)
        output_df = apply_restriction(output_df, restriction_path)

        diagnosis_df = output_df[variables_list]
        diagnosis_df.to_csv(path.join(results_path, 'AD.tsv'), sep='\t')
        sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
        logger.info('Found %s AD subjects for a total of %s sessions\n' % (len(sub_df), len(diagnosis_df)))

    if 'BV' in diagnoses:
        logger.info('Beginning the selection of BV label')
        output_df = stable_selection(bids_df, diagnosis='BV', logger=logger)
        output_df = mod_selection(output_df, missing_mods_dict, modality)
        output_df = apply_restriction(output_df, restriction_path)

        diagnosis_df = output_df[variables_list]
        diagnosis_df.to_csv(path.join(results_path, 'BV.tsv'), sep='\t')
        sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
        logger.info('Found %s BV subjects for a total of %s sessions\n' % (len(sub_df), len(diagnosis_df)))

    if 'CN' in diagnoses:
        logger.info('Beginning the selection of CN label')
        output_df = stable_selection(bids_df, diagnosis='CN', logger=logger)
        output_df = mod_selection(output_df, missing_mods_dict, modality)
        output_df = apply_restriction(output_df, restriction_path)

        diagnosis_df = output_df[variables_list]
        diagnosis_df.to_csv(path.join(results_path, 'CN.tsv'), sep='\t')
        sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
        logger.info('Found %s CN subjects for a total of %s sessions\n' % (len(sub_df), len(diagnosis_df)))

    if 'MCI' in diagnoses:
        logger.info('Beginning of the selection of MCI label')
        MCI_df = mci_stability(bids_df, 10 ** 4, logger=logger)  # Remove rMCI independently from time horizon
        output_df = diagnosis_removal(MCI_df, diagnosis_list=['rMCI'])
        output_df = mod_selection(output_df, missing_mods_dict, modality)
        output_df = apply_restriction(output_df, restriction_path)

        # Relabelling everything as MCI
        output_df.diagnosis = ['MCI'] * len(output_df)

        diagnosis_df = output_df[variables_list]
        diagnosis_df.to_csv(path.join(results_path, 'MCI.tsv'), sep='\t')
        sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
        logger.info('Found %s MCI subjects for a total of %s sessions\n' % (len(sub_df), len(diagnosis_df)))

    if 'sMCI' in diagnoses:
        logger.info('Beginning of the selection of sMCI label')
        time_MCI_df = mci_stability(bids_df, time_horizon, logger=logger)
        output_df = diagnosis_removal(time_MCI_df, diagnosis_list=['rMCI', 'pMCI'])
        output_df = output_df[output_df.diagnosis == 'sMCI']
        output_df = mod_selection(output_df, missing_mods_dict, modality)
        output_df = apply_restriction(output_df, restriction_path)

        diagnosis_df = output_df[variables_list]
        diagnosis_df.to_csv(path.join(results_path, 'sMCI.tsv'), sep='\t')
        sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
        logger.info('Found %s sMCI subjects for a total of %s sessions\n' % (len(sub_df), len(diagnosis_df)))

    if 'pMCI' in diagnoses:
        logger.info('Beginning of the selection of pMCI label')
        if time_MCI_df is None:
            time_MCI_df = mci_stability(bids_df, time_horizon)
        output_df = time_MCI_df[time_MCI_df.diagnosis == 'pMCI']
        output_df = mod_selection(output_df, missing_mods_dict, modality)
        output_df = apply_restriction(output_df, restriction_path)

        diagnosis_df = output_df[variables_list]
        diagnosis_df.to_csv(path.join(results_path, 'pMCI.tsv'), sep='\t')
        sub_df = diagnosis_df.reset_index().groupby('participant_id')['session_id'].nunique()
        logger.info('Found %s pMCI subjects for a total of %s sessions\n' % (len(sub_df), len(diagnosis_df)))
