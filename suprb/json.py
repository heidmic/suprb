import json
import numpy as np

from copy import deepcopy
from suprb.suprb import SupRB
from .solution import Solution

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
    json_config = {"elitist": {}, "config":  {}, "pool": []}

    try:
        suprb._cleanup()
    except AttributeError:
        pass
    _save_elitist(suprb, json_config)
    _save_config(suprb, json_config)
    _save_pool(suprb.pool_, json_config)
    # _save_input_space(suprb.pool_, json_config)

    with open(filename, "w") as f:
        json.dump(json_config, f)

    return json_config


def load(filename):
    with open(filename) as json_file:
        json_dict = json.load(json_file)

    suprb = _load_config(deepcopy(json_dict["config"]))
    _load_pool(json_dict, suprb)
    _load_elitist(suprb)
    suprb.is_fitted_ = True

    return suprb


def _save_elitist(suprb, json_config):
    json_config["elitist"] = {"complexity_": suprb.elitist_.complexity_,
                              "error_": suprb.elitist_.error_,
                              "fitness_": suprb.elitist_.fitness_}


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


def _save_config(suprb, json_config):
    json_config["config"] = {}
    primitive = (int, str, bool, float)

    for key, value in suprb.get_params().items():
        if key.startswith("logger"):
            continue
        elif isinstance(value, primitive):
            json_config["config"][key] = value
        else:
            json_config["config"][key] = _get_full_class_name(value)


def _save_pool(pool, json_config):
    json_config["pool"] = []

    for rule in pool:
        json_config["pool"].append(_convert_rule_to_json(rule))


def _save_input_space(pool, json_config):
    json_config["input_space"] = _convert_to_json_format(pool[0].input_space)


def _convert_dict_to_json(rule_dict):
    converted_dict = {}

    for key, value in rule_dict.items():
        converted_dict[key] = _convert_to_json_format(value)

    return converted_dict


def _convert_rule_to_json(rule):
    return {"error_":        rule.error_,
            "experience_":   rule.experience_,
            "match":         _convert_dict_to_json(vars(rule.match)),
            "is_fitted_":    rule.is_fitted_,
            "model":         {"coef_":          _convert_to_json_format(getattr(rule.model, "coef_")),
                              "intercept_":     getattr(rule.model, "intercept_")}}


def _load_config(json_config):
    _deserialize_config(json_config)
    return SupRB(**json_config)


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
        if json_config[longest_key] == "class:numpy.ndarray":
            json_config[longest_key] = np.ndarray([])
        else:
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


def _load_pool(json_dict, suprb):
    suprb.pool_ = []

    for rule in json_dict["pool"]:
        suprb.pool_.append(_convert_json_to_rule(rule, json_dict))


def _convert_json_to_rule(json_rule, json_dict):

    rule = Rule(_convert_matching_type(json_rule["match"], json_dict["config"]["matching_type"]),
                _convert_from_json_to_array(json_dict["input_space"]),
                _convert_model(json_rule["model"], json_dict["config"]["rule_generation__init__model"]),
                _get_class(json_dict["config"]["rule_generation__init__fitness"]))

    rule.error_ = json_rule["error_"]
    rule.experience_ = json_rule["experience_"]
    rule.is_fitted_ = json_rule["is_fitted_"]

    return rule


def _convert_matching_type(match, matching_type):
    matching = _get_class(matching_type)([])

    for name, p in match.items():
        setattr(matching, name, _convert_from_json_to_array(p))

    return matching


def _convert_model(json_model, model_type):
    model = _get_class(model_type)()

    setattr(model, "coef_", _convert_from_json_to_array(json_model["coef_"]))
    setattr(model, "intercept_", json_model["intercept_"])

    return model


def _load_elitist(suprb):
    suprb.elitist_ = Solution(genome=np.ones(len(suprb.pool_)),
                              pool=suprb.pool_,
                              mixing=suprb.solution_composition.init.mixing,
                              fitness=suprb.solution_composition.init.fitness)
