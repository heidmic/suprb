import warnings

from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm

"""
This is a wrapper class for SupRB, where parameters can be set without the need to create objects from deep within suprb.

An instance of SupRB gets instantiated with its default values. Depending on the parameters that are given to SupRBWrapper
the SupRB instance parameters are adapted. This way one can only set the parameters that they want to change, without the 
necessity to create the whole object. 

There are two possible ways to set parameters using SupRBWrapper (the below examples result in the same configuration):

1) Have a separate parameter for each nested parameter:
    - solution_composition__crossover=suprb.optimizer.solution.ga.crossover.NPoint()
    - solution_composition__crossover__crossover_rate=0.91
    - solution_composition__crossover__n=3

2) Create an object for a nested parameter and pass that object to the SuprbWrapper:
    - solution_composition__crossover=suprb.optimizer.solution.ga.crossover.NPoint(crossover_rate=0.91, n=3)

SupRBWrapper has some sanity checks to make sure that the parameter name is in fact part of SupRB. If it is not, it will issue a warning.

There is also the possibility to print the final configuration of the SupRB instance using the parameter print_config
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
