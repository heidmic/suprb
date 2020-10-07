from problems import amgauss, make_problem
from suprb2 import LCS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__ == '__main__':
    prob = make_problem("amgauss", 1)

    X, y = prob.generate(10000)

    y = np.reshape(y, (-1, 1))

    scale = StandardScaler()

    scale.fit(y)

    X_train, X_test, y_train, y_test = train_test_split(X, scale.transform(y))

    lcs = LCS(xdim=(prob.xdim+prob.adim))

    lcs.fit(X_train, y_train)

    error = mean_squared_error(y_test, lcs.predict(X_test))

