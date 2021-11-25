import os.path

import torch

from .singleton_pattern import Singleton


@Singleton
class GeneralSettings:

    """
    General settings, shared across the whole code.
    Singleton pattern.
    """

    ####################################################################################################################
    ### Core:
    ####################################################################################################################

    def __init__(self):
        self.dimension = 512
        self.output_dir = "output"
        self.preprocessing_dir = "preprocessing"
        #
        # # Whether or not to use the state file to resume the computation
        # self.load_state = False
        # # Default path to state file
        # self.state_file = os.path.join(self.output_dir, "pydef_state.p")
        #
        if torch.cuda.is_available():
            self.tensor_scalar_type = torch.cuda.FloatTensor
            self.tensor_integer_type = torch.cuda.DoubleTensor
        else:
            self.tensor_scalar_type = torch.FloatTensor
            self.tensor_integer_type = torch.DoubleTensor
        #
        # self.number_of_processes = 1
        #
        # self.dense_mode = False
        #
        # pydeformetrica_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        # self.unit_tests_data_dir = os.path.join(pydeformetrica_root, "tests", "unit_tests", "data")
        pass

    # def set_output_dir(self, output_dir):
    #     self.output_dir = output_dir
    #     self.state_file = os.path.join(output_dir, os.path.basename(self.state_file))

    ####################################################################################################################
    ### For multiprocessing:
    ####################################################################################################################

    # def serialize(self):
    #     return (self.dimension, self.output_dir, self.preprocessing_dir, self.load_state, self.state_file,
    #             self.tensor_scalar_type, self.tensor_integer_type, self.number_of_processes, self.dense_mode,
    #             self.unit_tests_data_dir)
    #
    # def initialize(self, args):
    #     (self.dimension, self.output_dir, self.preprocessing_dir, self.load_state, self.state_file,
    #      self.tensor_scalar_type, self.tensor_integer_type, self.number_of_processes, self.dense_mode,
    #      self.unit_tests_data_dir) = args


def Settings():
    return GeneralSettings.Instance()
