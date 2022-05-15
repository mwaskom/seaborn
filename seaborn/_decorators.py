from inspect import signature


def share_init_params_with_map(cls):
    """Make cls.map a classmethod with same signature as cls.__init__."""
    map_sig = signature(cls.map)
    init_sig = signature(cls.__init__)

    new = [v for k, v in init_sig.parameters.items() if k != "self"]
    new.insert(0, map_sig.parameters["cls"])
    cls.map.__signature__ = map_sig.replace(parameters=new)
    cls.map.__doc__ = cls.__init__.__doc__

    cls.map = classmethod(cls.map)

    return cls
