from problems import amgauss, make_problem
from suprb2 import LCS
from suprb2.random_gen import Random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def f(X):
    return 0.75*X**3-5*X**2+4*X+12


if __name__ == '__main__':
    n = 10000

    """prob = make_problem("amgauss", 1)

    X, y = prob.generate(n)

    y = np.reshape(y, (-1, 1))
    
    xdim = prob.xdim + prob.adim"""

    X = Random().random.uniform(-2.5, 7, (n, 1))
    y = f(X)
    xdim = 1

    scale_X = MinMaxScaler(feature_range=(-1, 1))
    scale_X.fit(X)
    X = scale_X.transform(X)

    scale_y = StandardScaler()
    scale_y.fit(y)
    y = scale_y.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, scale.transform(y), random_state=Random().split_seed())

    lcs = LCS(xdim=(prob.xdim+prob.adim), pop_size=50, ind_size=25, cl_min_range=0.2, generations=50)

    lcs.fit(X_train, y_train)

    error = mean_squared_error(y_test, lcs.predict(X_test))

