class Config():
    """
    We use Alex Martelli's Borg pattern to share our random generator properly
    in a kind of singleton way.

    See https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/ .
    """
    __shared_state = {"pop_size": 30,
                      "ind_size": 50,
                      "n_elitists": 1,
                      "generations": 5,
                      "xdim": None,
                      "default_prediction": None,
                      "var": None}

    def __init__(self):
        self.__dict__ = self.__shared_state


