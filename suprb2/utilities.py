import numpy as np

class Utilities:
  def default_error(y: np.ndarray):
      if y.size == 0:
          return 0
      else:
          # for standardised data this should be equivalent to np.var(y)
          return np.sum(y**2)/len(y)
