def create_list_hashes(path_folder, extensions_to_keep=(".nii.gz", ".tsv", ".json")):
    """
    Computes a dictionary of files with their corresponding hashes

        Args:
            (string) path_folder: starting point for the tree listing.
            (tuple) extensions_to_keep: files with these extensions will have their hashes computed and tracked

        Returns:
            (dictionary) all_files: a dictionary of the form {/path/to/file.extension: hash(file.extension)}
    """
    import hashlib
    import os

    def file_as_bytes(file):
        with file:
            return file.read()

    all_files = []
    for subdir, dirs, files in os.walk(path_folder):
        files.sort()
        for file in files:
            if file.lower().endswith(extensions_to_keep):
                all_files.append(os.path.join(subdir, file))

    dict_hashes = {
        fname[len(path_folder) :]: str(
            hashlib.md5(file_as_bytes(open(fname, "rb"))).digest()
        )
        for fname in all_files
    }
    return dict_hashes


def compare_folders_with_hashes(path_folder, list_hashes):
    """
    Compares the files of a folder against a reference

        Args:
            (string) path_folder: starting point for the tree listing.
            (dictionary) list_hashes: a dictionary of the form {/path/to/file.extension: hash(file.extension)}
    """
    import pickle

    hashes_check = pickle.load(open(list_hashes, "rb"))
    hashes_new = create_list_hashes(path_folder)

    if hashes_check != hashes_new:
        error_message1 = ""
        error_message2 = ""
        for key in hashes_check:
            if key not in hashes_new:
                error_message1 += "{0} not found !\n".format(key)
            elif hashes_check[key] != hashes_new[key]:
                error_message2 += "{0} does not match the reference file !\n".format(
                    key
                )
        raise ValueError(error_message1 + error_message2)
