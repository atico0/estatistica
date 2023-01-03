import numpy as np
import pandas as pd





class regressao_linear_simples(object):
  def __init__(self):
    pass
  
  def treinar(self, X, Y):
    mediax = np.mean(X)
    mediay = np.mean(Y)
    self.b2 = np.sum((X - mediax) * (Y - mediay)) / (np.sum((X - mediax)**2))
    self.b1 = mediay - self.b2 * mediax
    return

  def intercepto(self):
      return self.b1

  def coeficiente(self):
    return self.b2

  def previsao(self, x):
    return self.b1 + x * self.b2



class regressao_linear_multipla(object):
  def __init__(self):
    pass
  
  def treinar(self, X, Y):
    uns = np.ones((X.shape[0],1))
    self.X = np.concatenate((uns, X), axis=1)
    X_linha_X = np.linalg.multi_dot([np.matrix.transpose(self.X), self.X])
    X_linha_X_inverso = np.linalg.inv(X_linha_X)
    X_linha_y = np.linalg.multi_dot([np.matrix.transpose(self.X), y])
    self.b = np.linalg.multi_dot([X_linha_X_inverso, X_linha_y])

  def coeficientes(self):
    return self.b

  def previsao(self, x):
    return np.linalg.multi_dot([x, self.b])
