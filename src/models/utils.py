from src.models.dynamic_graph import DynamicGraph


def get_model(name, **kwargs):

    if name == "dynamic_graph":

        return DynamicGraph(**kwargs)
    
    else:
        raise ValueError("model not implemented: '{}'".format(name))
