def metric_config_equals(metrics1: list, metrics2: list, keys_to_compare=None):
    for metric1 in metrics1:
        if type(metric1) != type(metrics2):
            return False

    # Récupère tous les attributes non-callables, non-privés (exclut _xyz, methods, etc.)
    def get_public_attrs(metrics):
        return {
            k: v
            for k, v in vars(metrics).items()
            if not k.startswith("_") and not callable(v)
        }

    attrs1 = get_public_attrs(metrics1)
    attrs2 = get_public_attrs(metrics2)

    if keys_to_compare is not None:
        # Restreint la comparison à certains paramètres
        attrs1 = {k: attrs1.get(k) for k in keys_to_compare}
        attrs2 = {k: attrs2.get(k) for k in keys_to_compare}

    return attrs1 == attrs2
