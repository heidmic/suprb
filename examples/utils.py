def get_params(obj, prefix="", ignore_keywords=None):
    params = {}

    for attr_name, attr_value in obj.__dict__.items():
        if not any(keyword in attr_name for keyword in ignore_keywords):
            full_attr_name = f"{prefix}{attr_name}"

            if hasattr(attr_value, '__dict__'):
                params.update(get_params(attr_value, prefix=f"{full_attr_name}__", ignore_keywords=ignore_keywords))
            else:
                params[full_attr_name] = attr_value

    return params


def log_scores(scores):
    ignore_keywords = {
        "rule_generation_seeds_",
        "solution_composition_seeds_",
        "elitist___pool",
        "pool_",
        "random_state_",
        "elitist___mixing__filter_subpopulation__random_state",
        "elitist___genome",
        "solution_composition__init__mixing__filter_subpopulation__random_state"
    }

    for key, value in scores.items():
        if key == "estimator":
            params = get_params(value[0], ignore_keywords=ignore_keywords)
            for param, val in params.items():
                if not any(keyword in param for keyword in ignore_keywords):
                    print(f"{param}: {val}")
        else:
            print(f"{key}: {value}")
