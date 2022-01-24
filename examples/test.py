import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

from suprb2 import SupRB2
from suprb2 import rule
from suprb2.logging.stdout import StdoutLogger
from suprb2.optimizer.individual import ga
from suprb2.optimizer.rule import ns, es

if __name__ == '__main__':
    random_state = 42

    data, _ = fetch_openml(name='Concrete_Data', version=1, return_X_y=True)
    data = data.to_numpy()

    X, y = data[:, :8], data[:, 8]
    X, y = shuffle(X, y, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    models = [
        SupRB2(
            rule_generation=ns.NoveltySearch(
                n_iter=100,
                init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(alpha=0.8)),
                mutation=ns.mutation.HalfnormIncrease(sigma=2)
            ),
            individual_optimizer=ga.GeneticAlgorithm(
                n_iter=128,
                crossover=ga.crossover.Uniform(),
                selection=ga.selection.Tournament(),
                mutation=ga.mutation.BitFlips(),
            ),
            n_iter=16,
            n_rules=16,
            logger=StdoutLogger(),
            random_state=random_state,
        )
    ]
    models = {model.__class__.__name__: model for model in models}


    def run(name, model):
        print(f"[EVALUATION] {name}")
        return pd.Series(cross_val_score(model, X, y, cv=4, n_jobs=4, verbose=10, scoring='r2'), name='r2')


    scores = pd.concat({name: run(name=name, model=model) for name, model in models.items()}, axis=0).to_frame()
    scores.index.names = ['model', 'cv']

    print(scores.groupby(by='model').describe().to_string())