from click_option_group import OptionGroup

computational_group = OptionGroup(
    "Computational options", help="Context for runtime execution."
)
reproducibility_group = OptionGroup(
    "Reproducibility options", help="Allow to setup a deterministic setting."
)
model_group = OptionGroup(
    "Model options", help="Options allowing to choose the network trained."
)
data_group = OptionGroup("Data management", help="Options related to data management.")
cross_validation = OptionGroup(
    "Validation setup", help="Allow to choose the validation framework to use."
)
optimization_group = OptionGroup(
    "Optimization options", help="Options related to the optimizer."
)
transfer_learning_group = OptionGroup(
    "Transfer learning", help="Allow to choose a source MAPS."
)
task_group = OptionGroup(
    "Task specific options", help="Options specific to the task learnt by the network."
)
informations_group = OptionGroup(
    "Informative options",
    help=" Allow to get some more informations during the training.",
)
