from inspect import getmembers, isclass

import clinicadl.utils.network as network_package


def get_model_list(architecture=None, input_size=(128, 128)):
    """"""
    if not architecture:
        model_list = getmembers(network_package, isclass)
        for model in model_list:
            print(model[0])
    else:
        model_class = getattr(network_package, architecture)
        args = list(
            model_class.__init__.__code__.co_varnames[
                : model_class.__init__.__code__.co_argcount
            ]
        )
        args.remove("self")
        kwargs = dict()
        kwargs["input_size"] = (1, input_size[0], input_size[1])
        kwargs["use_cpu"] = True

        model = model_class(**kwargs)
        print(model.layers)
