from problems import amgauss, make_problem
from suprb2 import LCS
from suprb2.random_gen import Random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import numpy as np

def f(X):
    return 0.75*X**3-5*X**2+4*X+12


if __name__ == '__main__':
    print(f"Starting at {datetime.now().time()}")
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
    print(f"Samples generated. Starting training at {datetime.now().time()}")

    lcs = LCS(xdim=xdim, pop_size=50, ind_size=10, cl_min_range=0.2, generations=50,
              fitness="pseudo-BIC")

    lcs.fit(X_train, y_train)

    y_pred = lcs.predict(X_test)

    error = mean_squared_error(y_test, y_pred)
    print(f"Finished at {datetime.now().time()}. RMSE was {np.sqrt(error)}")

