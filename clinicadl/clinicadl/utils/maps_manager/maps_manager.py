import logging

level_dict = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class MapsManager:
    def __init__(self, maps_path, parameters=None, verbose="warning"):
        """
        Args:
            maps_path (str): path of the MAPS
            parameters (Dict[str:object]): parameters of the training step. If given a new MAPS is created.
            verbose (str): Logging level ("debug", "info", "warning", "error", "critical")
        """
        from logging import getLogger
        from os import listdir, path

        self.maps_path = maps_path
        if verbose not in level_dict:
            raise ValueError(
                f"Your verbose value {verbose} is incorrect."
                f"Please choose between the following values {level_dict.keys()}."
            )
        self.logger = getLogger("clinicadl")
        self.logger.setLevel(level_dict[verbose])

        # Existing MAPS
        if parameters is None:
            if not path.exists(path.join(maps_path, "maps.json")):
                raise ValueError(
                    f"MAPS was not found at {maps_path}."
                    f"To initiate a new MAPS please give a train_dict."
                )
            self.parameters = self._load_parameters()

        # Initiate MAPS
        else:
            if (path.exists(maps_path) and not path.isdir(maps_path)) or (
                path.isdir(maps_path) and not listdir(maps_path)
            ):
                raise ValueError(
                    f"You are trying a new MAPS at {maps_path} but"
                    f"this already corresponds to a file or a non-empty folder."
                    f"Please remove it or choose another location."
                )
            self.logger.info(f"A new MAPS was created at {maps_path}")
            self._check_args(parameters)
            self.parameters = parameters
            self._write_parameters()
            self._write_requirements_version()

    def _load_parameters(self):
        import json
        from os import path

        json_path = path.join(self.maps_path, "maps.json")
        with open(json_path, "r") as f:
            parameters = json.load(f)

        # Types of retro-compatibility
        # Change arg name: ex network --> model
        # Change arg value: ex for preprocessing: mni --> t1-extensive
        # New arg with default hard-coded value --> discarded_slice --> 20
        retro_change_name = {
            "network": "model",
            "mri_plane": "slice_direction",
            "pretrained_path": "transfer_learning_path",
            "pretrained_difference": "transfer_learning_difference",
            "patch_stride": "stride_size",
            "selection": "transfer_learning_selection",
        }
        retro_change_value = {
            "preprocessing": {"mni": "t1-extensive", "linear": "t1-linear"}
        }
        retro_add = {
            "discarded_slices": 20,
            "loss": "default",
            "uncropped_roi": False,
            "roi_list": None,
            "multi_cohort": False,
            "predict_atlas_intensities": None,  # To remove after multi-task implementation
            "merged_tsv_path": None,  # To remove after multi-task implementation
            "atlas_weight": 1,  # To remove after multi-task implementation
        }

        for old_name, new_name in retro_change_name.keys():
            if old_name in parameters:
                parameters[new_name] = parameters[old_name]
                del parameters[old_name]

        for name, change_values in retro_change_value.keys():
            if parameters[name] in change_values:
                parameters[name] = change_values[parameters[name]]

        for name, value in retro_add.keys():
            if name not in parameters:
                parameters[name] = value

        return parameters

    @staticmethod
    def _check_args(parameters):
        mandatory_arguments = [
            "caps_directory",
            "tsv_path",
            "preprocessing",
            "mode",
            "network_type",
            "model",
        ]

        for arg in mandatory_arguments:
            if arg not in parameters:
                raise ValueError(
                    f"The values of mandatory arguments {mandatory_arguments} should be set."
                    f"No value was given for {arg}."
                )

        # click passing context @click.command / @click.passcontext (config.json)
        # or default parameters in click --> from config_param import learning_rate --> @learning_rate

    def _write_parameters(self):
        import json
        from os import makedirs, path

        makedirs(self.maps_path, exist_ok=True)

        # save to json file
        json = json.dumps(self.parameters, skipkeys=True, indent=4)
        json_path = path.join(self.maps_path, "maps.json")
        self.logger.info(f"Path of json file: {json_path}")
        with open(json_path, "w") as f:
            f.write(json)

    def _write_requirements_version(self):
        import subprocess
        from os import path

        try:
            env_variables = subprocess.check_output("pip freeze", shell=True).decode(
                "utf-8"
            )
            with open(path.join(self.maps_path, "environment.txt"), "w") as file:
                file.write(env_variables)
        except subprocess.CalledProcessError:
            self.logger.warning(
                "You do not have the right to execute pip freeze. Your environment will not be written"
            )

    def __getattr__(self, name):
        """Allow to directly get the values in parameters attribute"""
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise AttributeError(f"'MapsManager' object has no attribute '{name}'")

    def __setattr__(self, key, value):
        if key in self.parameters:
            raise ValueError(
                f"{key} value cannot be changed."
                f"To train with a different value please create a new MAPS."
            )
        else:
            setattr(self, key, value)

    def train(self, fold_list=None):
        from clinicadl.train import train_autoencoder, train_multi_cnn, train_single_cnn

        if self.network_type == "autoencoder":
            self.parameters["transfer_learning_selection"] = "best_loss"
            train_autoencoder(self)
        elif self.network_type == "cnn":
            train_single_cnn(self)
        elif self.network_type == "multicnn":
            train_multi_cnn(self)
        else:
            raise NotImplementedError(
                f"Framework {self.network_type} not implemented in clinicadl"
            )
