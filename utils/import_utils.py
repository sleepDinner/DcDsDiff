import importlib
from functools import partial

import omegaconf
import inspect


def instantiate_from_config(config, target_key="name", target_params_key="params", **kwargs):
    """
    Instantiate an object from a config dict.
    Args:
        config (dict, omegaconf.dictconfig.DictConfig): Config dict.
        target_key (str): Key of the target object.
        target_params_key (str): Key of the target object's parameters.
    Returns:
        object: Instantiated object.
    """
    if not isinstance(config, dict):
        config = dict(config)
    params = dict(config.get(target_params_key, dict()))
    # kwargs 的优先级高于 YAML params，常用于把 model 或 optimizer params 注入进去。
    merge_params = {**params, **kwargs}
    # config["name"] 是完整导入路径，例如 model.net.net 或 torch.optim.AdamW。
    return get_obj_from_str(config[target_key])(**merge_params)


def recurse_instantiate_from_config(config, target_key="name", target_params_key="params", **kwargs):
    """
    Recursively instantiate an object from a config dict.
    Args:
        config (dict, omegaconf.dictconfig.DictConfig): Config dict.
        target_key (str): Key of the target object.
        target_params_key (str): Key of the target object's parameters.
    Returns:
        object: Instantiated object.
    """
    if not isinstance(config, dict):
        config = dict(config)
    params = dict(config.get(target_params_key, dict()))

    for k, v in params.items():
        if isinstance(v, dict) or isinstance(v, omegaconf.dictconfig.DictConfig):
            # 如果某个参数本身也是一个带 name/params 的配置，就先递归实例化它。
            params[k] = recurse_instantiate_from_config(v, target_key, target_params_key)
    merge_params = {**params, **kwargs}
    return get_obj_from_str(config[target_key])(**merge_params)


def get_obj_from_str(string, reload=False):
    """
    Get an object from a string.
    Args:
        string (str): String of the object.
        reload (bool): Whether to reload the module.
    Returns:
        Class: Class of the object.
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 动态导入模块并取出类/函数对象，后续再由调用方实例化或调用。
    return getattr(importlib.import_module(module, package=None), cls)


class ClassInstance:
    """
    Create a class instance that can be called to create multiple instances of the target class.
    Use this class to create a class instead of a instance of a class, when you want to create in
    recurse_instantiate_from_config function.
    """

    def __new__(cls, target: str or type, **kwargs):
        if isinstance(target, str):
            target = get_obj_from_str(target)
        elif isinstance(target, type):
            pass
        else:
            raise TypeError(f'Invalid type for target_class: {type(target)}')
        return partial(target, **kwargs)
        # return target


def fill_args_from_dict(func, args_dict):
    args = inspect.getfullargspec(func).args
    # 只取目标函数签名中存在的字段，避免 batch 中额外 key 导致调用报错。
    args_dict = {k: v for k, v in args_dict.items() if k in args}
    return partial(func, **args_dict)
