import warnings

from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm

"""
solution_composition__crossover=suprb.optimizer.solution.ga.crossover.NPoint(crossover_rate=0.91, n=3),

solution_composition__crossover=suprb.optimizer.solution.ga.crossover.NPoint(),
solution_composition__crossover__crossover_rate=0.91,
solution_composition__crossover__n=3,
"""
class SupRBWrapper():
    def __new__(self, **kwargs):
        self.suprb = SupRB(rule_generation=ES1xLambda(),
                  solution_composition=GeneticAlgorithm())
        
        kwargs = dict(sorted(kwargs.items(), key=lambda item: item[0].count("__")))
        
        for key, value in kwargs.items():
            if "print_config" == key:
                continue

            attribute_string = key.replace("__", ".")
            attribute_split = attribute_string.split(".")
            attribute_value = self.suprb

            for attribute in attribute_split[:-1]:
                if hasattr(attribute_value, attribute):
                    attribute_value = getattr(attribute_value, attribute)
                
            if hasattr(attribute_value, attribute_split[-1]):
                setattr(attribute_value, attribute_split[-1], value)
            else:
                warning_text = "The config has conflicting parameters!"
                warning_text += f"\n{attribute_string} is not part of the config and can have negative effects on the execution!\n\n"
                warnings.warn(warning_text)

        if "print_config" in kwargs and kwargs["print_config"]:
            for attr_name, attr_value in self.suprb.__dict__.items():
                print(attr_name, attr_value)

        return self.suprb
