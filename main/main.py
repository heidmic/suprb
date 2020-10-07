from problems import amgauss, make_problem
from suprb2 import LCS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    prob = make_problem("amgauss", 1)

    X, y = prob.generate(25000)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lcs = LCS()

    lcs.fit(X_train, y_train)

    error = mean_squared_error(y_test, lcs.predict(X_test))

