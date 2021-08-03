import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ranksums, shapiro
from scipy.stats import t

datasets = ["auto_sweden", "haberman", "multidim_cubic", "banknote", "wine_quality_reg", "wine_quality_clas", "communities", "sonar"]
optimizers = ["CMA", "MLSP", "ML"]
optimizer_tuples = [["CMA", "MLSP"], ["CMA", "ML"], ["MLSP", "ML"]]
metrics = ["time", "rmse", "f1_score", "diversity"]

def evaluate():
    with open("final_data/evaluation.txt", "a") as f:
        for ds in datasets:
            basic_infos(f, ds)
            f.write(f"  Statical Tests:\n")
            for met in metrics:
                if not os.path.isfile(f"final_data/{ds}/ML/{met}.csv"):
                    continue

                f.write(f"    {met}:\n")
                shapiro_dict = shapiro_test(f, ds, met)
                statistical_test(f, shapiro_dict)
            f.write(f"\n")


def basic_infos(file, dataset):
    file.write(f"{dataset}:\n")
    for opt in optimizers:
        file.write(f"  {opt}:\n")
        for met in metrics:
            file_path = f"final_data/{dataset}/{opt}/{met}.csv"
            if os.path.isfile(file_path):
                data = pd.read_csv(file_path, sep=',', header=None).values[:,-1]
                file.write(f"    {met}:\n")
                file.write(f"      Mean:      {np.mean(data):.9f}\n")
                file.write(f"      Std. dev.: {np.std(data):.9f}\n")
                file.write(f"      Median:    {np.median(data):.9f}\n")


def shapiro_test(file, dataset, metric):
    cma_data = pd.read_csv(f"final_data/{dataset}/CMA/{metric}.csv", sep=',', header=None).values[:,-1]
    mlsp_data = pd.read_csv(f"final_data/{dataset}/MLSP/{metric}.csv", sep=',', header=None).values[:,-1]
    ml_data = pd.read_csv(f"final_data/{dataset}/ML/{metric}.csv", sep=',', header=None).values[:,-1]
    _, cma_shapiro = shapiro(cma_data)
    _, mlsp_shapiro = shapiro(mlsp_data)
    _, ml_shapiro = shapiro(ml_data)
    file.write(f"      Shaphiro p-values:\n")
    file.write(f"        CMA:  {cma_shapiro:.9f}\n")
    file.write(f"        MLSP: {mlsp_shapiro:.9f}\n")
    file.write(f"        ML:   {ml_shapiro:.9f}\n")
    return {"CMA": [cma_shapiro, cma_data], "MLSP": [mlsp_shapiro, mlsp_data], "ML": [ml_shapiro, ml_data]}


def statistical_test(file, shapiro_dict):
    for opt_tuple in optimizer_tuples:
        data_1 = shapiro_dict[opt_tuple[0]]
        data_2 = shapiro_dict[opt_tuple[1]]

        # Correspondent statistical test
        if data_1[0] < 0.05 and data_2[0] < 0.05:
            _, pvalue = ttest_ind(data_1[1], data_2[1], equal_var=(np.var(data_1[1]) == np.var(data_2[1])))
            file.write(f"      {opt_tuple[0]} and {opt_tuple[1]} Student's t-test p-value: {pvalue:.9f}\n")
        else:
            _, pvalue = ranksums(data_1[1], data_2[1])
            file.write(f"      {opt_tuple[0]} and {opt_tuple[1]} Wilcoxon rank-sum test p-value: {pvalue:.9f}\n")


if __name__ == '__main__':
    evaluate()
