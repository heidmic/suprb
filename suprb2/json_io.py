import json
import numpy as np

import suprb2
from suprb2.suprb2 import SupRB2
from .individual import Individual
from suprb2.optimizer.rule import es
from suprb2.optimizer.individual import ga
from suprb2.optimizer.individual.archive import Elitist
from suprb2.individual.mixing_model import ErrorExperienceHeuristic
from suprb2.individual.fitness import ComplexityWu
from sklearn.linear_model import LinearRegression
from suprb2.logging.stdout import StdoutLogger
from suprb2.logging.mlflow import MlflowLogger
from suprb2.logging.combination import CombinedLogger

from .rule import Rule
from suprb2 import rule
import importlib


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class JsonIO:
    def __init__(self):
        self.suprb = None
        self.json_config = {"config": None, "pool": [], "elitist": None}

    def save(self, suprb, filename):
        suprb._cleanup()
        self._save_config(suprb)
        self._save_pool(suprb.pool_)

        with open(filename, "w") as f:
            json.dump(self.json_config, f)

    def load(self, filename):
        with open(filename) as json_file:
            json_dict = json.load(json_file)

        self._load_config(json_dict["config"])
        self._load_pool(json_dict["pool"])
        self._load_elitist()
        self.suprb.is_fitted_ = True

        return self.suprb

    # Static Helper Functions
    @staticmethod
    def convert_to_json_format(np_array):
        return json.dumps(np_array, cls=NumpyEncoder)

    @staticmethod
    def convert_from_json_format(json_array):
        return np.array(json.loads(json_array))

    # Save Config
    def _save_config(self, suprb):
        params = {}
        primitive = (int, str, bool, float)

        for key, value in suprb.get_params().items():
            if key.startswith("logger"):
                continue
            elif isinstance(value, primitive):
                params[key] = value
            else:
                params[key] = self._get_full_class_name(value)

        self.json_config["config"] = params

    def _get_full_class_name(self, instance):
        class_name = instance.__class__
        module = class_name.__module__
        if module == 'builtins':
            return class_name.__qualname__
        return "class:" + module + '.' + class_name.__qualname__

    # Save Pool
    def _save_pool(self, pool):
        for rule in pool:
            self.json_config["pool"].append(self._convert_rule_to_json(rule))

    def _convert_rule_to_json(self, rule):
        return {"error_":        rule.error_,
                "experience_":   rule.experience_,
                "input_space":   JsonIO.convert_to_json_format(rule.input_space),
                "bounds":        JsonIO.convert_to_json_format(rule.bounds),
                "fitness":       self._get_full_class_name(rule.fitness),
                "is_fitted_":    rule.is_fitted_,
                "model":         self._get_json_model(rule.model)}

    def _get_json_model(self, model):
        model_params = {"coef_":          JsonIO.convert_to_json_format(getattr(model, "coef_")),
                        "rank_":          getattr(model, "rank_"),
                        "singular_":      JsonIO.convert_to_json_format(getattr(model, "singular_")),
                        "n_features_in_": getattr(model, "n_features_in_"),
                        "intercept_":     getattr(model, "intercept_")}

        return {self._get_full_class_name(model): {"init_params": model.get_params(),
                                                   "model_params": model_params}}

    # Load Config
    def _load_config(self, json_config):
        self._deserialize_config(json_config)
        json_config["logger"] = CombinedLogger([('stdout', StdoutLogger()), ('mlflow', MlflowLogger())])
        self.suprb = SupRB2(**json_config)

    def _deserialize_config(self, json_config):
        number_of_arguments_for_suprb = 8

        while len(list(json_config.keys())) != number_of_arguments_for_suprb:
            base_key, longest_key = self._get_longest_key(json_config)
            self._update_longest_key(json_config, longest_key)
            params = self._update_same_base_keys(json_config, base_key)
            json_config[base_key] = self._get_class(json_config[base_key])(**params)

    def _get_longest_key(self, json_config):
        longest_key = max(json_config, key=lambda key: key.count('__'))
        base_key = '__'.join(longest_key.split('__')[:-1])

        return base_key, longest_key

    def _update_longest_key(self, json_config, longest_key):
        if isinstance(json_config[longest_key], str) and json_config[longest_key].startswith("class:"):
            json_config[longest_key] = self._get_class(json_config[longest_key])()
        else:
            json_config[longest_key] = json_config[longest_key]

    def _get_class(self, full_class_name):
        full_class_name = full_class_name[6:]
        separator_idx = full_class_name.rfind(".")
        module_string = full_class_name[:separator_idx]
        class_string = full_class_name[separator_idx+1:]

        module = importlib.import_module(module_string)
        return getattr(module, class_string)

    def _update_same_base_keys(self, json_config, base_key):
        params = {}

        for key in self._get_keys_with_same_base(json_config, base_key):
            param = key[key.rfind("__") + 2:]

            if json_config[key] == "NoneType":
                params[param] = None
            elif isinstance(json_config[key], str) and json_config[key].startswith("class:"):
                json_config[key] = self._get_class(json_config[key])()
            else:
                params[param] = json_config[key]

            del json_config[key]

        return params

    def _get_keys_with_same_base(self, json_config, base):
        same_base_key_list = []
        for key in json_config:
            if key.startswith(base) and key != base:
                same_base_key_list.append(key)

        return same_base_key_list

    # Load Pool

    def _load_pool(self, json_pool):
        self.suprb.pool_ = []

        for rule in json_pool:
            self.suprb.pool_.append(self._convert_json_to_rule(rule))

    def _convert_json_to_rule(self, json_rule):
        rule = Rule(
            JsonIO.convert_from_json_format(json_rule["bounds"]),
            JsonIO.convert_from_json_format(json_rule["input_space"]),
            self._convert_model(json_rule["model"]),
            self._get_class(json_rule["fitness"]))

        rule.error_ = json_rule["error_"]
        rule.experience_ = json_rule["experience_"]
        rule.is_fitted_ = json_rule["is_fitted_"]

        return rule

    def _convert_model(self, json_model):
        model_name = list(json_model.keys())[0]
        model = self._get_class(model_name)(**json_model[model_name]["init_params"])

        for name, p in json_model[model_name]["model_params"].items():
            if (name == "coef_") or (name == "singular_"):
                setattr(model, name, JsonIO.convert_from_json_format(p))
            else:
                setattr(model, name, p)

        return model

        # Load Elitist
    def _load_elitist(self):
        self.suprb.elitist_ = Individual(genome=np.ones(len(self.suprb.pool_)),
                                         pool=self.suprb.pool_,
                                         mixing=self.suprb.individual_optimizer.init.mixing,
                                         fitness=self.suprb.individual_optimizer.init.fitness)
