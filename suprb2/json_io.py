import json
import numpy as np
from suprb2.individual import mixing_model

from suprb2.suprb2 import SupRB2
from .individual import Individual
from suprb2.optimizer.rule import es
from suprb2.optimizer.individual import ga
from suprb2.individual.mixing_model import ErrorExperienceHeuristic
from suprb2.individual.fitness import ComplexityWu
from sklearn.linear_model import LinearRegression


from .rule import Rule
from suprb2 import rule


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
        self._save_config(suprb)
        self._save_pool(suprb.pool_)
        self._save_elitist(suprb.elitist_)

        with open(filename, "w") as f:
            json.dump(self.json_config, f)

    def load(self, filename):
        with open(filename) as json_file:
            json_dict = json.load(json_file)

        self._load_config(json_dict["config"])
        self._load_pool(json_dict["pool"])
        self._load_elitist(json_dict["elitist"])
        self.suprb.is_fitted_ = True

        return self.suprb

    # Static Helper Functions
    @staticmethod
    def convert_to_json_format(np_array):
        return json.dumps(np_array, cls=NumpyEncoder)

    @staticmethod
    def convert_from_json_format(json_array):
        return np.array(json.loads(json_array))

    @staticmethod
    def convert_key_value(string_json):
        if string_json.startswith("ComplexityWu"):
            return ComplexityWu()
        elif string_json.startswith("ErrorExperienceHeuristic"):
            return ErrorExperienceHeuristic()
        elif string_json.startswith("HalfnormIncrease"):
            return es.mutation.HalfnormIncrease()
        elif string_json.startswith("MeanInit(fitness=VolumeWu()"):
            return rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu())
        elif string_json.startswith("Uniform"):
            return ga.crossover.Uniform()
        elif string_json.startswith("Tournament"):
            return ga.selection.Tournament()

    @staticmethod
    def convert_individual_to_json(rule):
        return {"error_":        rule.error_,
                "fitness_":      rule.fitness_,
                "fitness":       str(rule.fitness),
                "is_fitted_":    rule.is_fitted_}

    # Save Config
    def _save_config(self, suprb):
        self.json_config["config"] = {"n_iter":                 suprb.n_iter,
                                      "n_initial_rules":        suprb.n_initial_rules,
                                      "n_rules":                suprb.n_rules,
                                      "rule_generation":        str(suprb.rule_generation),
                                      "individual_optimizer":   str(suprb.individual_optimizer),
                                      "random_state":           suprb.random_state,
                                      "verbose":                suprb.verbose,
                                      "logger":                 str(suprb.logger),
                                      "n_jobs":                 suprb.n_jobs}

    # Save Pool
    def _save_pool(self, pool):
        for rule in pool:
            self.json_config["pool"].append(self._convert_rule_to_json(rule))

    def _convert_rule_to_json(self, rule):
        return {"error_":        rule.error_,
                "experience_":   rule.experience_,
                "input_space":   JsonIO.convert_to_json_format(rule.input_space),
                "bounds":        JsonIO.convert_to_json_format(rule.bounds),
                "fitness_":      rule.fitness_,
                "match_":        JsonIO.convert_to_json_format(rule.match_),
                "pred_":         JsonIO.convert_to_json_format(rule.pred_),
                "fitness":       str(rule.fitness),
                "is_fitted_":    rule.is_fitted_,
                "model":         self._get_json_model(rule.model)}

    def _get_json_model(self, model):
        return {str(model): {"init_params": model.get_params(),
                             "model_params": {"coef_":          JsonIO.convert_to_json_format(getattr(model, "coef_")),
                                              "rank_":          getattr(model, "rank_"),
                                              "singular_":      JsonIO.convert_to_json_format(getattr(model, "singular_")),
                                              "n_features_in_": getattr(model, "n_features_in_"),
                                              "intercept_":     getattr(model, "intercept_")}}}

    # Save Elitist
    def _save_elitist(self, elitist):
        self.json_config["elitist"] = {"genome":        JsonIO.convert_to_json_format(elitist.genome),
                                       "mixing":        str(elitist.mixing),
                                       "fitness":       str(elitist.fitness),
                                       "individual":    JsonIO.convert_individual_to_json(elitist)}

    # Load Config
    def _load_config(self, json_config):
        self.suprb = SupRB2()
        self.suprb.set_params(**json_config)

    def _convert_model(self, json_model):
        for model_name in json_model:
            if model_name.startswith("LinearRegression"):
                model = LinearRegression(**json_model[model_name]["init_params"])
                for name, p in json_model[model_name]["model_params"].items():
                    if (name == "coef_") or (name == "singular_"):
                        setattr(model, name, JsonIO.convert_from_json_format(p))
                    else:
                        setattr(model, name, p)

                return model

    # Load Elitist
    def _load_elitist(self, json_elitist):
        self.suprb.elitist_ = Individual(genome=JsonIO.convert_from_json_format(json_elitist["genome"]),
                                         pool=self.suprb.pool_,
                                         mixing=JsonIO.convert_key_value(json_elitist["mixing"]),
                                         fitness=JsonIO.convert_key_value(json_elitist["fitness"]))

        self.suprb.elitist_.error_ = json_elitist["individual"]["error_"]
        self.suprb.elitist_.fitness_ = json_elitist["individual"]["fitness_"]
        self.suprb.elitist_.fitness = JsonIO.convert_key_value(json_elitist["individual"]["fitness"])
        self.suprb.elitist_.is_fitted_ = json_elitist["individual"]["is_fitted_"]

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
            json_rule["fitness"])

        rule.error_ = json_rule["error_"]
        rule.experience_ = json_rule["experience_"]
        rule.fitness_ = json_rule["fitness_"]
        rule.match_ = JsonIO.convert_from_json_format(json_rule["match_"])
        rule.pred_ = JsonIO.convert_from_json_format(json_rule["pred_"])
        rule.is_fitted_ = json_rule["is_fitted_"]

        return rule
