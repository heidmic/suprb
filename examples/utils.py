def log_scores(scores):
    for key, value in scores.items():
        if key == "estimator":
            for attr_name, attr_value in value[0].__dict__.items():
                if "seed" not in attr_name and "pool_" != attr_name:
                    try:
                        print(attr_name, attr_value)
                    except:
                        print("None value found")
        else:
            print(key, value)
