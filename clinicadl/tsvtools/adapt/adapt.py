
import os
from logging import getLogger
import pandas as pd
import json
import os
from typing import Any, Dict

import toml

from clinicadl.prepare_data.prepare_data_utils import compute_folder_and_file_type

from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError

logger = getLogger("clinicadl")

def concat_files(file_path, df_baseline, df_all) :
    label_df = pd.read_csv(file_path, sep="\t")
    if file_path.endswith("baseline.tsv") :
        df_baseline = pd.concat([df_baseline, label_df])
    elif file_path.endswith(".tsv") :
        df_all = pd.concat([df_all, label_df])
    return df_all, df_baseline

def adapt_split(files_list_2, path, new_tsv_dir):
    df_split_baseline = pd.DataFrame()
    df_split_all = pd.DataFrame()
    print("adapt_split")
    print(f"files list 2 : {files_list_2}")
    print(f"path: {path}")
    print(f"new_tsv_dir: {new_tsv_dir}")
    for file2 in files_list_2 :
        path_2 = os.path.join(path, file2)
        if file2.endswith(".tsv"):
            df_split_all, df_split_baseline = concat_files(path_2, df_split_baseline, df_split_all)

    
    df_split_all.to_csv(os.path.join(new_tsv_dir, "labels.tsv"), sep="\t")
    if not df_split_baseline.empty :
        df_split_baseline.to_csv(os.path.join(new_tsv_dir, "labels_baseline.tsv"), sep="\t")

def adapt_kfold(files_list_2, path, new_tsv_dir):
    df_split_baseline = pd.DataFrame()
    df_split_all = pd.DataFrame()
    print("adapt_kfold")
    for file2 in files_list_2 :
        path_2 = os.path.join(path, file2)
        print(f"path_2 = path + file2 : {path_2}")
        if file2.endswith(".tsv"):
            df_split_all, df_split_baseline = concat_files(path_2, df_split_baseline, df_split_all)
        elif os.path.isdir(path_2):
            df_fold_all = pd.DataFrame()
            df_kfold_baseline = pd.DataFrame()
            new_path_2 = os.path.join(new_tsv_dir, file2)
            os.makedirs(new_path_2)
            print(f"new path 2 = {new_path_2}")
            files_list_3 = os.listdir(path_2)    
            for file3 in files_list_3 :
                path_3= os.path.join(path_2, file3)
                print(f"path3= {path_3}")
                if os.path.isdir(path_3):
                    files_list_4 = os.listdir(path_3)
                    new_path_3 = os.path.join(new_tsv_dir, file2,file3)
                    os.makedirs(new_path_3)
                    for file4 in files_list_4 :
                        path_4= os.path.join(path_3, file4)
                        print(f"path4= {path_4}")
                        if file4.endswith(".tsv"):
                            df_fold_all, df_kfold_baseline = concat_files(path_4, df_kfold_baseline, df_fold_all)
                    df_fold_all.to_csv(os.path.join(new_path_3, "labels.tsv"), sep="\t")
                    if not df_kfold_baseline.empty :
                        df_kfold_baseline.to_csv(os.path.join(new_path_3, "labels_baseline.tsv"), sep="\t")

                    #df_fold_all, df_kfold_baseline = concat_files(path_3, df_kfold_baseline, df_fold_all)
        
                
        df_split_all.to_csv(os.path.join(new_tsv_dir, "labels.tsv"), sep="\t")
        if not df_split_baseline.empty :
            df_split_baseline.to_csv(os.path.join(new_tsv_dir, "labels_baseline.tsv"), sep="\t")



def adapt(old_tsv_dir : str, new_tsv_dir : str, labels_list ):


    if not os.path.exists(old_tsv_dir):
        raise ClinicaDLArgumentError( 
            f"the directory: {old_tsv_dir} doesn't exists"
            "Please give an existing path"
        )
    if os.path.exists(new_tsv_dir):
        raise ClinicaDLArgumentError( 
            f"the directory: {new_tsv_dir} already exists"
            "Please give a new name to the folder"
        )
    
    os.makedirs(new_tsv_dir, exist_ok=True)

    files_list = os.listdir(old_tsv_dir)
    print(f"old_dir ({old_tsv_dir}) contains: files list :{files_list}")

    df_baseline = pd.DataFrame()
    df_all = pd.DataFrame()

    for file in files_list :
        path = os.path.join(old_tsv_dir, file)
        print(path)
        if os.path.isfile(path) and path.endswith(".tsv"):
            df_all, df_baseline = concat_files(path, df_baseline, df_all)

        elif os.path.isdir(path):
            print("babababab")
            print(os.path.join(new_tsv_dir, file))
            os.makedirs(os.path.join(new_tsv_dir, file), exist_ok=True)
            files_list_2 = os.listdir(path)
            print(files_list_2)
            if "split.json" in files_list :
                adapt_split(files_list_2, path, os.path.join(new_tsv_dir, file))

            if "kfold.json" in files_list_2:
                adapt_kfold(files_list_2, path, os.path.join(new_tsv_dir, file))

            # with open(os.path.join(old_tsv_dir + "split.json"), "r") as f:
            #     parameters = json.load(f)
            # for label in labels_list : 
            #     tsv_path = os.path.join(old_tsv_dir, label + ".tsv")

        # elif "kfold.json" in files_list :
        #     print("ok")
       
        # else :
        #     raise ClinicaDLArgumentError(
        #         f"The given directory ({old_tsv_dir}) doesn't contain a split.json or a kfold.json file"
        #     )
    if not df_all.empty :
        df_all.to_csv(os.path.join(new_tsv_dir, "labels.tsv"), sep="\t")
    if not df_baseline.empty:
        df_baseline.to_csv(os.path.join(new_tsv_dir, "labels_baseline.tsv"), sep="\t")

    # for (root, directories, files) in os.walk("/Users/camille.brianceau/aramis/clinicadl_tuto/tsvtools_finish/labels_list"):
    #     print(root)
    #     print(directories)
    #     print(files)
    #     df_baseline = pd.DataFrame()
    #     df_all = pd.DataFrame()
    #     if not (len(files)==0):
    #         for file in files:
    #             file_path = os.path.join(root, file)
    #             if "split.json" in files :
    #                 df_all, df_baseline = concat_files(file_path, df_baseline, df_all)
                    
    #             elif "kfold.json" in files :
    #                 df_all, df_baseline = concat_files(file_path, df_baseline, df_all)

                



