import json
import numpy as np

from suprb2.suprb2 import SupRB2
from .individual import Individual
from suprb2.logging.stdout import StdoutLogger
from suprb2.logging.mlflow import MlflowLogger
from suprb2.logging.combination import CombinedLogger

from .rule import Rule
import importlib

"""(De)Serialization currenlty only supports LinearRegression as local models. """

CLASS_PREFIX = "class:"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(obj)


def dump(suprb, filename):
    json_config = {"config": None, "pool": []}

    suprb._cleanup()
    json_config = _save_config(suprb)
    _save_pool(suprb.pool_, json_config)

    with open(filename, "w") as f:
        json.dump(json_config, f)

    return json_config


def load(filename):
    with open(filename) as json_file:
        json_dict = json.load(json_file)

    suprb = _load_config(json_dict["config"])
    _load_pool(json_dict["pool"], suprb)
    _load_elitist(suprb)
    suprb.is_fitted_ = True

    return suprb


def _convert_to_json_format(np_array):
    return json.dumps(np_array, cls=NumpyEncoder)


def _convert_from_json_to_array(json_array):
    return np.array(json.loads(json_array))


def _get_full_class_name(instance):
    class_name = instance.__class__
    module = class_name.__module__
    if module == 'builtins':
        return class_name.__qualname__
    return CLASS_PREFIX + module + '.' + class_name.__qualname__


def _save_config(suprb):
    json_config = {}
    json_config["config"] = {}
    primitive = (int, str, bool, float)

    for key, value in suprb.get_params().items():
        if key.startswith("logger"):
            continue
        elif isinstance(value, primitive):
            json_config["config"][key] = value
        else:
            json_config["config"][key] = _get_full_class_name(value)

    return json_config


def _save_pool(pool, json_config):
    json_config["pool"] = []

    for rule in pool:
        json_config["pool"].append(_convert_rule_to_json(rule))


def _convert_rule_to_json(rule):
    return {"error_":        rule.error_,
            "experience_":   rule.experience_,
            "input_space":   _convert_to_json_format(rule.input_space),
            "bounds":        _convert_to_json_format(rule.bounds),
            "fitness":       _get_full_class_name(rule.fitness),
            "is_fitted_":    rule.is_fitted_,
            "model":         _convert_linear_regression_to_json(rule.model)}


def _convert_linear_regression_to_json(model):
    model_params = {"coef_":          _convert_to_json_format(getattr(model, "coef_")),
                    "rank_":          getattr(model, "rank_"),
                    "singular_":      _convert_to_json_format(getattr(model, "singular_")),
                    "n_features_in_": getattr(model, "n_features_in_"),
                    "intercept_":     getattr(model, "intercept_")}

    return {_get_full_class_name(model): {"model_params": model_params}}


def _load_config(json_config):
    _deserialize_config(json_config)
    return SupRB2(**json_config)


def _deserialize_config(json_config):
    while "__" in "".join(json_config.keys()):
        base_key, longest_key = _get_longest_key(json_config)
        _update_longest_key(json_config, longest_key)
        params = _update_same_base_keys(json_config, base_key)
        json_config[base_key] = _get_class(json_config[base_key])(**params)


def _get_longest_key(json_config):
    longest_key = max(json_config, key=lambda key: key.count('__'))
    base_key = '__'.join(longest_key.split('__')[:-1])

    return base_key, longest_key


def _update_longest_key(json_config, longest_key):
    if isinstance(json_config[longest_key], str) and json_config[longest_key].startswith(CLASS_PREFIX):
        json_config[longest_key] = _get_class(json_config[longest_key])()
    else:
        json_config[longest_key] = json_config[longest_key]


def _get_class(full_class_name):
    full_class_name = full_class_name.replace(CLASS_PREFIX, "")
    separator_idx = full_class_name.rfind(".")
    module_string = full_class_name[:separator_idx]
    class_string = full_class_name[separator_idx+1:]

    module = importlib.import_module(module_string)
    return getattr(module, class_string)


def _update_same_base_keys(json_config, base_key):
    params = {}

    for key in _get_keys_with_same_base(json_config, base_key):
        param = key[key.rfind("__") + 2:]

        if json_config[key] == "NoneType":
            params[param] = None
        elif isinstance(json_config[key], str) and json_config[key].startswith(CLASS_PREFIX):
            json_config[key] = _get_class(json_config[key])()
        else:
            params[param] = json_config[key]

        del json_config[key]

    return params


def _get_keys_with_same_base(json_config, base):
    same_base_key_list = []
    for key in json_config:
        if key.startswith(base) and key != base:
            same_base_key_list.append(key)

    return same_base_key_list


def _load_pool(json_pool, suprb):
    suprb.pool_ = []

    for rule in json_pool:
        suprb.pool_.append(_convert_json_to_rule(rule))


def _convert_json_to_rule(json_rule):
    rule = Rule(
        _convert_from_json_to_array(json_rule["bounds"]),
        _convert_from_json_to_array(json_rule["input_space"]),
        _convert_model(json_rule["model"]),
        _get_class(json_rule["fitness"]))

    rule.error_ = json_rule["error_"]
    rule.experience_ = json_rule["experience_"]
    rule.is_fitted_ = json_rule["is_fitted_"]

    return rule


def _convert_model(json_model):
    model_name = list(json_model.keys())[0]
    model = _get_class(model_name)()

    for name, p in json_model[model_name]["model_params"].items():
        if (name == "coef_") or (name == "singular_"):
            setattr(model, name, _convert_from_json_to_array(p))
        else:
            setattr(model, name, p)

    return model


def _load_elitist(suprb):
    suprb.elitist_ = Individual(genome=np.ones(len(suprb.pool_)),
                                pool=suprb.pool_,
                                mixing=suprb.individual_optimizer.init.mixing,
                                fitness=suprb.individual_optimizer.init.fitness)
