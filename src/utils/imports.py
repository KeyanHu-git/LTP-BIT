import importlib


def get_obj_from_str(path: str, reload: bool = False):
    module, name = path.rsplit(".", 1)
    if reload:
        module_obj = importlib.import_module(module)
        importlib.reload(module_obj)
    return getattr(importlib.import_module(module, package=None), name)
