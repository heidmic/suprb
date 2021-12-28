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

from suprb2 import individual


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
                "input_space":   self._convert_to_json_format(rule.input_space),
                "bounds":        self._convert_to_json_format(rule.bounds),
                "fitness_":      rule.fitness_,
                "match_":        self._convert_to_json_format(rule.match_),
                "pred_":         self._convert_to_json_format(rule.pred_),
                "fitness":       str(rule.fitness),
                "is_fitted_":    rule.is_fitted_,
                "model":         self._get_json_model(rule.model)}

    def _get_json_model(self, model):
        mp = {}

        for p in ("coef_", "rank_", "singular_", "n_features_in_", "intercept_"):
            if (p == "coef_") or (p == "singular_"):
                mp[p] = self._convert_to_json_format(getattr(model, p))
            else:
                mp[p] = getattr(model, p)

        return {str(model): {"init_params": model.get_params(),
                             "model_params": mp}}

    # Save Elitist
    def _save_elitist(self, elitist):
        self.json_config["elitist"] = {"genome":        self._convert_to_json_format(elitist.genome),
                                       "mixing":        str(elitist.mixing),
                                       "fitness":       str(elitist.fitness),
                                       "individual":    self._convert_individual_to_json(elitist)}

    def _convert_individual_to_json(self, rule):
        return {"error_":        rule.error_,
                "fitness_":      rule.fitness_,
                "fitness":       str(rule.fitness),
                "is_fitted_":    rule.is_fitted_}

    # Load Config
    def _load_config(self, json_config):
        rule_generation = self._get_rule_generation_from_json(json_config)
        individual_optimizer = self._get_individual_optimizer_from_json(json_config)

        self.suprb = SupRB2(
            rule_generation=rule_generation,
            individual_optimizer=individual_optimizer,
            n_iter=json_config["n_iter"],
            n_initial_rules=json_config["n_initial_rules"],
            n_rules=json_config["n_rules"],
            random_state=json_config["random_state"],
            verbose=json_config["verbose"],
            n_jobs=json_config["n_jobs"])

    # Load Rule Generation
    def _get_rule_generation_from_json(self, json_config):
        rule_generation = json_config["rule_generation"]

        if rule_generation.startswith("ES1xLambda"):
            n_iter = self._get_value_from_string(rule_generation, "n_iter")
            operator = self._get_value_from_string(rule_generation, "operator")
            init = self._convert_key_value(self._get_value_from_string(rule_generation, "init"))
            mutation = self._convert_key_value(self._get_value_from_string(rule_generation, "mutation"))

            return es.ES1xLambda(
                n_iter=int(n_iter),
                operator=operator,
                init=init,
                mutation=mutation)

    # Load Individual Optimizer
    def _get_individual_optimizer_from_json(self, json_config):
        individual_optimizer = json_config["individual_optimizer"]

        if individual_optimizer.startswith("GeneticAlgorithm"):
            n_iter = self._get_value_from_string(individual_optimizer, "n_iter")
            crossover = self._convert_key_value(self._get_value_from_string(individual_optimizer, "crossover"))
            selection = self._convert_key_value(self._get_value_from_string(individual_optimizer, "selection"))

            return ga.GeneticAlgorithm(
                n_iter=int(n_iter),
                crossover=crossover,
                selection=selection
            )

    def _convert_json_to_rule(self, json_rule):
        rule = Rule(
            self._convert_from_json_format(json_rule["bounds"]),
            self._convert_from_json_format(json_rule["input_space"]),
            self._convert_model(json_rule["model"]),
            json_rule["fitness"])

        rule.error_ = json_rule["error_"]
        rule.experience_ = json_rule["experience_"]
        rule.fitness_ = json_rule["fitness_"]
        rule.match_ = self._convert_from_json_format(json_rule["match_"])
        rule.pred_ = self._convert_from_json_format(json_rule["pred_"])
        rule.is_fitted_ = json_rule["is_fitted_"]

        return rule

    def _convert_model(self, json_model):
        for model_name in json_model:
            if model_name.startswith("LinearRegression"):
                model = LinearRegression(**json_model[model_name]["init_params"])
                for name, p in json_model[model_name]["model_params"].items():
                    if (name == "coef_") or (name == "singular_"):
                        setattr(model, name, self._convert_from_json_format(p))
                    else:
                        setattr(model, name, p)

                return model

    # Load Elitist
    def _load_elitist(self, json_elitist):
        self.suprb.elitist_ = Individual(genome=self._convert_from_json_format(json_elitist["genome"]),
                                         pool=self.suprb.pool_,
                                         mixing=self._convert_key_value(json_elitist["mixing"]),
                                         fitness=self._convert_key_value(json_elitist["fitness"]))

        self.suprb.elitist_.error_ = json_elitist["individual"]["error_"]
        self.suprb.elitist_.fitness_ = json_elitist["individual"]["fitness_"]
        self.suprb.elitist_.fitness = self._convert_key_value(json_elitist["individual"]["fitness"])
        self.suprb.elitist_.is_fitted_ = json_elitist["individual"]["is_fitted_"]

    # Load Pool
    def _load_pool(self, json_pool):
        self.suprb.pool_ = []

        for rule in json_pool:
            self.suprb.pool_.append(self._convert_json_to_rule(rule))

    # Helper Functions
    def _get_value_from_string(self, string, param):
        start_idx = string.find(param) + len(param) + 1
        end_idx = string[start_idx:].find(",")

        if end_idx < 0:
            end_idx = string[start_idx:].find(")")

        return string[start_idx:start_idx + end_idx]

    def _convert_to_json_format(self, np_array):
        return json.dumps(np_array, cls=NumpyEncoder)

    def _convert_from_json_format(self, json_array):
        return np.array(json.loads(json_array))

    def _convert_key_value(self, string_json):
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
