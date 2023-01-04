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
  def __init__(self, X, Y, sigma=0):
    self.treinar(X,Y)
    self.estima_erro()
    if sigma:
       self.calcula_cov(sigma) 
    else:
      self.estima_var()
      self.estima_cov()
     
  def adiciona_uns(self, X):
    uns = np.ones((X.shape[0],1))
    return np.concatenate((uns, X), axis=1)
  
  def treinar(self, X, Y):
    self.X = X
    self.X_com_uns = self.adiciona_uns(X)
    self.X_linha_X = np.linalg.multi_dot([np.matrix.transpose(self.X_com_uns), self.X_com_uns])
    self.X_linha_X_inv = np.linalg.inv(self.X_linha_X)
    self.X_linha_y = np.linalg.multi_dot([np.matrix.transpose(self.X_com_uns), y])
    self.beta_hat = np.linalg.multi_dot([self.X_linha_X_inv, self.X_linha_y])

  def coeficientes(self):
    return self.beta_hat

  def previsao(self, x):
    x = self.adiciona_uns(x)
    return np.linalg.multi_dot([x, self.beta_hat])
  
  def estima_erro(self):
    self.e_hat = y - self.previsao(self.X)
    return self.e_hat
  
  def estima_var(self):
    self.sigma_hat = np.sum(self.e_hat**2) / (self.X.shape[0] - self.X.shape[1])
    return self.sigma_hat

  def estima_cov(self):
    self.cov_hat = self.sigma_hat*self.X_linha_X_inv
    return self.cov_hat
  
  def calcula_cov(self, sigma=0):
    if sigma:
      self.sigma = sigma
      self.cov = sigma*self.X_linha_X_inv
      return self.cov
    print('PARA CALCULAR A REAL COVARIÂNCI, É NECESSÁRIO O VALOR REAL DA VARIÂNCIA')
  

  def estima_int(self, alpha):

    intervalos = []
    if self.sigma:
      diagonal = np.diag(self.cov)
      z = sts.norm().cdf(1 - (alpha/2))
      for i in range(len(self.beta_hat)):
        intervalos.append([self.beta_hat[i] - z*diagonal[i], self.beta_hat[i] + z*diagonal[i]])
      return intervalos

    diagonal = np.diag(self.cov_hat)
    df = self.X_com_uns.shape[0] - self.X_com_uns.shape[1]
    t = sts.chi2(df).cdf(1 - (alpha/2))
    for i in range(len(self.beta_hat)):
      intervalos.append([self.beta_hat[i] - t*diagonal[i], self.beta_hat[i] + t*diagonal[i]])
    return intervalos
    
  def teste_t(self, j, beta_j):
    t = (self.beta_hat[j] - beta_j) / (self.cov_hat[j,j]**0.5)
    print(f'{self.beta_hat[j]} - {beta_j} / {self.cov_hat[j,j]**0.5}')
    print('Estatística de teste: ',t)
    return t
    
        
