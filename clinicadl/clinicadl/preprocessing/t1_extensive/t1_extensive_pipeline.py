# coding: utf8

import clinica.pipelines.engine as cpe

# Use hash instead of parameters for iterables folder names
from nipype import config

cfg = dict(execution={"parameterize_dirs": False})
config.update_config(cfg)


class T1Extensive(cpe.Pipeline):
    """T1Extensive pipeline.

    Returns:
        A clinica pipeline object containing the T1Extensive pipeline.
    """

    @staticmethod
    def get_processed_images(caps_directory, subjects, sessions):
        import os
        from clinica.utils.filemanip import extract_image_ids
        from clinica.utils.inputs import clinica_file_reader

        information = {
            "pattern": os.path.join(
                "t1_extensive",
                "*_*_space-Ixi549Space_desc-SkullStripped_T1w.nii*",
            ),
            "description": "Skull-stripped T1w in Ixi549Space space.",
            "needed_pipeline": "t1-volume-tissue-segmentation",
        }
        image_ids = []
        if os.path.isdir(caps_directory):
            skull_stripped_files = clinica_file_reader(
                subjects, sessions, caps_directory, information, False
            )
            image_ids = extract_image_ids(skull_stripped_files)
        return image_ids

    def check_pipeline_parameters(self):
        """Check pipeline parameters."""
        pass

    def check_custom_dependencies(self):
        """Check dependencies that can not be listed in the `info.json` file."""
        pass

    def get_input_fields(self):
        """Specify the list of possible inputs of this pipelines.

        Returns:
            A list of (string) input fields name.
        """
        return ["norm_t1w"]

    def get_output_fields(self):
        """Specify the list of possible outputs of this pipelines.

        Returns:
            A list of (string) output fields name.
        """
        return ["skull_stripped_t1w"]

    def build_input_node(self):
        """Build and connect an input node to the pipeline."""
        import os
        import nipype.pipeline.engine as npe
        import nipype.interfaces.utility as nutil
        from clinica.utils.inputs import clinica_file_reader
        from clinica.utils.exceptions import ClinicaException
        from clinica.utils.stream import cprint
        from clinica.utils.ux import print_images_to_process

        all_errors = []
        t1w_in_ixi549space = {
            "pattern": os.path.join(
                "t1",
                "spm",
                "segmentation",
                "normalized_space",
                "*_*_space-Ixi549Space_T1w.nii*",
            ),
            "description": "Tissue probability map in native space",
            "needed_pipeline": "t1-volume-tissue-segmentation",
        }
        try:
            t1w_files = clinica_file_reader(
                self.subjects,
                self.sessions,
                self.caps_directory,
                t1w_in_ixi549space,
            )
        except ClinicaException as e:
            all_errors.append(e)

        # Raise all errors if some happened
        if len(all_errors) > 0:
            error_message = "Clinica faced errors while trying to read files in your CAPS directory.\n"
            for msg in all_errors:
                error_message += str(msg)
            raise RuntimeError(error_message)

        if len(self.subjects):
            print_images_to_process(self.subjects, self.sessions)
            cprint("The pipeline will last a few seconds per image.")

        read_node = npe.Node(
            name="ReadingFiles",
            iterables=[
                ("norm_t1w", t1w_files),
            ],
            synchronize=True,
            interface=nutil.IdentityInterface(fields=self.get_input_fields()),
        )
        self.connect(
            [
                (read_node, self.input_node, [("norm_t1w", "norm_t1w")]),
            ]
        )

    def build_output_node(self):
        """Build and connect an output node to the pipeline."""
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe
        import nipype.interfaces.io as nio
        from clinica.utils.nipype import fix_join, container_from_filename

        # Find container path from filename
        # =================================
        container_path = npe.Node(
            nutil.Function(
                input_names=["bids_or_caps_filename"],
                output_names=["container"],
                function=container_from_filename,
            ),
            name="container_path",
        )

        # Write results into CAPS
        # =======================
        write_node = npe.Node(name="write_results", interface=nio.DataSink())
        write_node.inputs.base_directory = self.caps_directory
        write_node.inputs.parameterization = False

        self.connect(
            [
                (self.input_node, container_path, [("norm_t1w", "bids_or_caps_filename")]),
                (self.output_node, write_node, [("skull_stripped_t1w", "t1_extensive.@skull_stripped_t1w")]),
                (container_path, write_node, [(("container", fix_join, ""), "container")]),
            ]
        )

    def build_core_nodes(self):
        """Build and connect the core nodes of the pipeline."""
        import os
        import nipype.pipeline.engine as npe
        import nipype.interfaces.utility as nutil
        from clinica.utils.inputs import RemoteFileStructure
        from .t1_extensive_utils import (
            get_caps_filename,
            apply_binary_mask,
            get_file_from_server
        )

        # Get CAPS Filename
        # =================
        caps_filename = npe.Node(
            name="0-GetCapsFilename",
            interface=nutil.Function(
                input_names="norm_t1w",
                output_names=self.get_output_fields(),
                function=get_caps_filename,
            ),
        )

        # Apply brainmask
        # ===============
        ICV_MASK = RemoteFileStructure(
            filename="tpl-IXI549Space_desc-ICV_mask.nii.gz",
            url="https://aramislab.paris.inria.fr/files/data/masks/tpl-IXI549Space/",
            checksum="1daebcae52218d48e4bd79328754d2e6415f80331c8b87f39ed289c4f4ec810a",
        )

        skull_stripping = npe.Node(
            name="1-SkullStripping",
            interface=nutil.Function(
                input_names=["input_img", "binary_img", "output_filename"],
                output_names=["masked_image_path"],
                function=apply_binary_mask,
            ),
        )
        skull_stripping.inputs.binary_img = get_file_from_server(
            ICV_MASK, os.path.join("clinicadl", "t1-extensive")
        )

        # Connection
        # ==========
        self.connect(
            [
                (self.input_node, caps_filename, [("norm_t1w", "norm_t1w")]),
                (self.input_node, skull_stripping, [("norm_t1w", "input_img")]),
                (caps_filename, skull_stripping, [("skull_stripped_t1w", "output_filename")]),
                (skull_stripping, self.output_node, [("masked_image_path", "skull_stripped_t1w")]),
            ]
        )
