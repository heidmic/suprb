import pandas as pd
import optuna
import os
import requests
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from zipfile import ZipFile
from optuna.trial import BaseTrial
import suprb.optimizer.rule.mutation
from suprb import SupRB
from suprb import rule
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.rule.origin import SquaredError
from suprb.optimizer.solution import saga1, saga2
from suprb.optimizer.rule import es

def download(url: str, dest_folder: str, filename: str):
    if not os.path.exists(dest_folder):
          os.makedirs(dest_folder)  # create folder if it does not exist

    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
          with open(file_path, 'wb') as f:
               for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                         f.write(chunk)
                         f.flush()
                         os.fsync(f.fileno())

def fetch_ccs():
     filename = 'concrete+compressive+strength.zip'
     try:
          zip_file = ZipFile('experiments/datasets/' + filename)
     except FileNotFoundError:
          download('http://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip', 'experiments/datasets', filename)
          zip_file = ZipFile('experiments/datasets/' + filename)

     xls_file = zip_file.open('Concrete_Data.xls')
     data = pd.read_excel(xls_file)
     return data

def fetch_ccpp():
     filename = 'combined+cycle+power+plant.zip'
     try:
          zip_file = ZipFile('experiments/datasets/' + filename)
     except FileNotFoundError:
          download('http://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip', 'experiments/datasets', filename)
          zip_file = ZipFile('experiments/datasets/' + filename)

     xlsx_file = zip_file.open('CCPP/Folds5x2_pp.xlsx')
     data = pd.read_excel(xlsx_file)
     return data

def fetch_asn():
     filename = 'airfoil+self+noise.zip'
     try:
          zip_file = ZipFile('experiments/datasets/' + filename)
     except FileNotFoundError:
          download('https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip', 'experiments/datasets', filename)
          zip_file = ZipFile('experiments/datasets/' + filename)

     dat_file = zip_file.open('airfoil_self_noise.dat')
     data = pd.read_csv(dat_file, delimiter='\t', names=['Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity', 'Suction side displacement thickness', 'Scaled sound pressure level'])
     return data

def fetch_pppts():
     filename = 'physicochemical+properties+of+protein+tertiary+structure.zip'
     try:
          zip_file = ZipFile('experiments/datasets/' + filename)
     except FileNotFoundError:
          download('https://archive.ics.uci.edu/static/public/265/physicochemical+properties+of+protein+tertiary+structure.zip', 'experiments/datasets', filename)
          zip_file = ZipFile('experiments/datasets/' + filename)

     csv_file = zip_file.open('CASP.csv')
     data = pd.read_csv(csv_file)
     return data

# Has two target columns
def fetch_pt():
     filename = 'parkinsons+telemonitoring.zip'
     try:
          zip_file = ZipFile('experiments/datasets/' + filename)
     except FileNotFoundError:
          download('https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip', 'experiments/datasets', filename)
          zip_file = ZipFile('experiments/datasets/' + filename)

     data_file = zip_file.open('parkinsons_updrs.data')
     data = pd.read_csv(data_file)
     return data


class Saga1Objective(object):

    def __init__(self, X, y, random_state):
        self.random_state = random_state
        self.X = X
        self.y = y

    def __call__(self, trial: BaseTrial):
        # Rule Discovery
        es_n_iter = 250

        es_lambda = 20

        es_delay = 25

        es_mutation_sigma = trial.suggest_float('es_mutation_sigma', 0.0, 3.0, step=0.01)

        es_mutation_operator = trial.suggest_categorical('es_mutation_operator', [suprb.optimizer.rule.mutation.Normal,
                                                                                  suprb.optimizer.rule.mutation.HalfnormIncrease,
                                                                                  suprb.optimizer.rule.mutation.UniformIncrease])
        es_init_operator = trial.suggest_categorical('es_init_operator', [rule.initialization.NormalInit,
                                                                          rule.initialization.MeanInit])
        es_alpha = trial.suggest_float('es_alpha', 0.0, 0.1, step=0.001)
        if es_init_operator == rule.initialization.NormalInit:
            es_init_sigma = trial.suggest_float('es_init_sigma', 0.0, 3.0, step=0.01)
            es_init_operator = es_init_operator(fitness=rule.fitness.VolumeWu(alpha=es_alpha), sigma=es_init_sigma)
        else:
            es_init_operator = es_init_operator(fitness=rule.fitness.VolumeWu(alpha=es_alpha))   
                 
        es_operator = trial.suggest_categorical('es_operator', ['+', '&', ','])
        if es_operator == '&':
            es_delay = trial.suggest_int('es_&_delay', 0, 25)
        else:
            es_n_iter = trial.suggest_int('es_iter', 0, 50)

        # Solution Composition
        # For Saga1 mutation_rate is not needed as a parameter, as is crossover_rate
        # n_iter is tested, as it is unclear how long self-adaption needs to take effect
        ga_n_iter = trial.suggest_int('ga_n_iter', 32, 256, step=32)

        ga_elitist_ratio = 0.17

        ga_population_size = 32

        ga_selection_operator = trial.suggest_categorical('ga_selection_operator', [saga1.selection.RouletteWheel, 
                                                                                    saga1.selection.Tournament, 
                                                                                    saga1.selection.LinearRank, 
                                                                                    saga1.selection.Random])
        if ga_selection_operator == saga1.selection.Tournament:
            ga_tourn_k = trial.suggest_int('ga_tourn_k', 1, 10)
            ga_selection_operator = ga_selection_operator(k=ga_tourn_k)
        else:
            ga_selection_operator = ga_selection_operator()

        ga_crossover_operator = trial.suggest_categorical('ga_crossover_operator', [saga1.crossover.NPoint, saga1.crossover.Uniform])
        if ga_crossover_operator == saga1.crossover.NPoint:
            ga_npoint_n = trial.suggest_int('ga_npoint_n', 1, 10)
            ga_crossover_operator = ga_crossover_operator(n=ga_npoint_n)
        else:
            ga_crossover_operator = ga_crossover_operator()

        supRB_with_saga1 = SupRB(
            rule_generation=es.ES1xLambda(
                n_iter=es_n_iter,
                origin_generation=SquaredError(),
                lmbda=es_lambda,
                delay=es_delay,
                operator=es_operator,
                init=es_init_operator,
                mutation=es_mutation_operator(sigma=es_mutation_sigma)
            ),
            solution_composition=saga1.SelfAdaptingGeneticAlgorithm(
                n_iter=ga_n_iter,
                elitist_ratio=ga_elitist_ratio,
                population_size=ga_population_size,
                crossover=ga_crossover_operator,
                selection=ga_selection_operator
            ),
            n_iter=32,
            n_rules=4,
            logger=StdoutLogger(),
            random_state=random_state,
            )
        score = cross_val_score(supRB_with_saga1, self.X, self.y, n_jobs=4, cv=4)
        return score
    

class Saga2Objective(object):

    def __init__(self, X, y, random_state):
        self.random_state = random_state
        self.X = X
        self.y = y

    def __call__(self, trial: BaseTrial):
        # Rule Discovery
        es_n_iter = 250

        es_lambda = 20

        es_delay = 25

        es_mutation_sigma = trial.suggest_float('es_mutation_sigma', 0.0, 3.0, step=0.01)

        es_mutation_operator = trial.suggest_categorical('es_mutation_operator', [suprb.optimizer.rule.mutation.Normal,
                                                                                  suprb.optimizer.rule.mutation.HalfnormIncrease,
                                                                                  suprb.optimizer.rule.mutation.UniformIncrease])
        es_init_operator = trial.suggest_categorical('es_init_operator', [rule.initialization.NormalInit,
                                                                          rule.initialization.MeanInit])
        es_alpha = trial.suggest_float('es_alpha', 0.0, 0.1, step=0.001)
        if es_init_operator == rule.initialization.NormalInit:
            es_init_sigma = trial.suggest_float('es_init_sigma', 0.0, 3.0, step=0.01)
            es_init_operator = es_init_operator(fitness=rule.fitness.VolumeWu(alpha=es_alpha), sigma=es_init_sigma)
        else:
            es_init_operator = es_init_operator(fitness=rule.fitness.VolumeWu(alpha=es_alpha))   
                 
        es_operator = trial.suggest_categorical('es_operator', ['+', '&', ','])
        if es_operator == '&':
            es_delay = trial.suggest_int('es_&_delay', 0, 25)
        else:
            es_n_iter = trial.suggest_int('es_iter', 0, 50)

        # Solution Composition
        # For Saga2 mutation_rate is not needed as a parameter, as is crossover_rate
        # n_iter is tested, as it is unclear how long self-adaption needs to take effect
        ga_n_iter = trial.suggest_int('ga_n_iter', 32, 256, step=32)

        ga_elitist_ratio = 0.17

        ga_population_size = 32

        ga_selection_operator = trial.suggest_categorical('ga_selection_operator', [saga2.selection.RouletteWheel, 
                                                                                    saga2.selection.Tournament, 
                                                                                    saga2.selection.LinearRank, 
                                                                                    saga2.selection.Random])
        if ga_selection_operator == saga1.selection.Tournament:
            ga_tourn_k = trial.suggest_int('ga_tourn_k', 1, 10)
            ga_selection_operator = ga_selection_operator(k=ga_tourn_k)
        else:
            ga_selection_operator = ga_selection_operator()

        ga_crossover_operator = trial.suggest_categorical('ga_crossover_operator', [saga2.crossover.NPoint, saga2.crossover.Uniform])
        if ga_crossover_operator == saga2.crossover.NPoint:
            ga_npoint_n = trial.suggest_int('ga_npoint_n', 1, 10)
            ga_crossover_operator = ga_crossover_operator(n=ga_npoint_n)
        else:
            ga_crossover_operator = ga_crossover_operator()

        supRB_with_saga2 = SupRB(
            rule_generation=es.ES1xLambda(
                n_iter=es_n_iter,
                origin_generation=SquaredError(),
                lmbda=es_lambda,
                delay=es_delay,
                operator=es_operator,
                init=es_init_operator,
                mutation=es_mutation_operator(sigma=es_mutation_sigma)
            ),
            solution_composition=saga2.SelfAdaptingGeneticAlgorithm(
                n_iter=ga_n_iter,
                elitist_ratio=ga_elitist_ratio,
                population_size=ga_population_size,
                crossover=ga_crossover_operator,
                selection=ga_selection_operator
            ),
            n_iter=32,
            n_rules=4,
            logger=StdoutLogger(),
            random_state=random_state,
            )
        score = cross_val_score(supRB_with_saga2, self.X, self.y, n_jobs=4, cv=4)
        return score
    

if __name__ == '__main__':
    random_state = 42

    css_data = fetch_ccs()
    cpp_data = fetch_ccpp()
    asn_data = fetch_asn()
    pppts_data = fetch_pppts().sample(frac=0.2, random_state=random_state)
    pt_data = fetch_pt()

    #CSS/SAGA1 STUDY
    css_data = css_data.to_numpy()
    X, y = css_data[:, :8], css_data[:, 8]
    X, y = shuffle(X, y, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))
    objective = Saga1Objective(X, y, random_state)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=24*3600)
    study.best_trial