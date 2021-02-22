class ClassifierPool:
    """
        We use Alex Martelli's Borg pattern to share our pool properly
        in a kind of singleton way.

        See https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/ .
        """
    __shared_state = {"classifiers": list()}

    def __init__(self,):
        self.__dict__ = self.__shared_state




