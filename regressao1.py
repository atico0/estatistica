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
  """Classe de regressão linear multipla"""

  def __init__(self, X, Y, sigma=0, intercepto=True):
    """
    X: matriz numérica
    matriz com as observações das variáveis explicativas
    
    Y: vetor numérica
    vetor com as observações da variável resposta

    sigma: valor numérico. sigma > 0
    valor da variância
    se o sigma não for dado, será usado o estimador de sigma

    intercepto: valor boleano
    se intercepto == True, a regressão é feita com o intercepto
    """

    self.intercepto = intercepto
    self.treinar(X,Y)
    self.T, self.K = self.X_com_uns.shape
    self.estima_erro()
    self.residuo_padronizados()
    self.residuo_studentizado()
    self.residuo_press()
    self.residuo_Rstudent()
    if sigma:
       self.calcula_cov(sigma) 
    else:
      self.estima_var()
      self.estima_cov()
    self.y_hat = self.previsao(self.X)
    self.coef_det()
    if intercepto:
      self.coef_ajt()
    
    print(f'R2: {self.R2}')
    print(f'R2 ajustado: {self.R2_ajt}')
    #print(f'F: {self.teste_F()}')

  
     

  def adiciona_uns(self, X):
    """
    Função que adiciona uma coluna de uns na matriz X e retorna essa nova matriz
    
    X: matriz numérico
    matriz em que será adicionada a coluna de uns
    """

    if self.intercepto:
      uns = np.ones((X.shape[0],1))
      return np.concatenate((uns, X), axis=1)
    return X
  
  def treinar(self, X, Y):
    """
    Função que Estima os coeficientes
    
    X: matriz numérica
    matriz com as observações das variáveis explicativas que será usada na estimação dos parâmetros
    
    Y: vetor numérica
    vetor com as observações da variável resposta que será usada na estimação dos parâmetros
    """

    self.X = X
    self.Y = Y
    self.X_com_uns = self.adiciona_uns(X)
    self.X_linha_X = np.linalg.multi_dot([np.matrix.transpose(self.X_com_uns), self.X_com_uns])
    self.X_linha_X_inv = np.linalg.inv(self.X_linha_X)
    self.X_linha_y = np.linalg.multi_dot([np.matrix.transpose(self.X_com_uns), self.Y])
    self.beta_hat = np.linalg.multi_dot([self.X_linha_X_inv, self.X_linha_y])


  def previsao(self, x):
    """
    Fazendo previsoes para uma matriz X
    
    x: matriz da qual serão feitas as previsões usando os parâmetros estimados
    """

    if self.intercepto:
      x = self.adiciona_uns(x)
    return np.linalg.multi_dot([x, self.beta_hat])
  
  def estima_erro(self):
    """
    Calulando os residuos com base nos X e Y fornecidos
    """

    self.e_hat = self.Y - self.previsao(self.X)
    return self.e_hat
  
  def residuo_padronizados(self):
    """
    Calulando os residuos padronizados
    """
    
    self.msres = np.sum(self.e_hat**2) / (self.T - self.K)
    self.d = self.e_hat / (self.msres**0.5)
    return self.d

  def residuo_studentizado(self):
    """
    Calulando os residuos studentizados
    """

    self.H = np.linalg.multi_dot([self.X_com_uns, self.X_linha_X_inv, np.matrix.transpose(self.X_com_uns)])
    self.H_diag = np.diag(self.H)
    self.r = self.e_hat / (self.msres * (1 - self.H_diag))**0.5
    return self.r

  def residuo_press(self):
    """
    Calulando os residuos PRESS
    """

    self.p = self.e_hat / (1 - self.H_diag)
    return self.p

  def residuo_Rstudent(self):
    """
    Calulando os residuos Rstudentizados
    """

    self.S2 = ((self.T - self.K) * self.msres - ((self.e_hat ** 2) / (1 - self.H_diag))) / (self.T - self.K - 1)
    self.t = self.e_hat / (self.S2 * (1 - self.H_diag))**0.5

  def estima_var(self):
    """
    Estimando o sigma 2 (para o caso em que sigma 2 não é fornecido)
    """

    self.sigma_hat = np.sum(self.e_hat**2) / (self.X.shape[0] - self.X.shape[1])
    return self.sigma_hat

  def estima_cov(self):
    """
    Estimando a matriz de covariâncias (com o valor estimado de sigma 2)
    """

    self.cov_hat = self.sigma_hat*self.X_linha_X_inv
    return self.cov_hat
  
  def calcula_cov(self, sigma=0):
    """
    Calculando a matriz de covariâncias (com o valor real de sigma 2)
    """
    if sigma:
      self.sigma = sigma
      self.cov = sigma*self.X_linha_X_inv
      return self.cov
    print('PARA CALCULAR A REAL COVARIÂNCIA, É NECESSÁRIO O VALOR REAL DA VARIÂNCIA')
  

  def estima_int(self, alpha):
    """
    Estima intervalos de confiança para os coeficientes
    
    alpha: valor numérico, 0 < alpha < 1
    nível de significância do alpha (quando menor o alpha maior a amplitude do intervalo)
    """
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
    """
    calculando a estatística do teste t
    Esse teste testa as hipoteses: 
    H0: beta[j] = beta_j contra H1: beta[j] != beta_j

    j: valor inteiro
    Indice do beta que será testado

    beta_j: valor numérico
    """

    t = (self.beta_hat[j] - beta_j) / (self.cov_hat[j,j]**0.5)
    print(f'{self.beta_hat[j]} - {beta_j} / {self.cov_hat[j,j]**0.5}')
    print('Estatística de teste: ',t)
    return t

  def test_F(self):
    """
    calculando a estatística do teste F
    Esse teste testa as hipoteses: 
    H0: beta[0] = beta[1] = ... = beta[k-1] =  0 contra H1: pelo menos 1 dos betas é diferente de 0
    """
    self.beta_hats = self.beta_hat[1:]
    self.cov_hat_s = self.cov_hat[1:,1:]
    beta_cov_inv_beta = np.linalg.multi_dot([np.matrix.transpose(self.beta_hats), np.linalg.inv(self.cov_hat_s), self.beta_hats])
    # np.linalg.multi_dot funciona como o np.dot e o np.matmul mas sever pra mais de duas matrizes por vez
    F = (beta_cov_inv_beta)/ (self.K - 1)
    return F

  def test_wald(self, R, r):
    """calculando a estatística do teste wald
    Esse teste testa as hipoteses: 
    H0: R * beta = r contra H1: R * beta != r
    
    R: matriz numérica
    matriz de restrições

    r: vetor númerico
    vetor de restrições
    """
    Rbeta_hatr = np.dot(R, self.beta_hat) - r
    trambolho = np.linalg.inv(self.sigma_hat * (np.linalg.multi_dot([R, self.X_linha_X_inv, np.matrix.transpose(R)])))
    w = np.linalg.multi_dot(Rbeta_hatr, trambolho, np.matrix.transpose(Rbeta_hatr))
    return w

  def coef_det(self):
    """
    Calculando o coeficiente de determinação (R2) que é a capacidade explicativa do modelo
    """
    if self.intercepto:
      media_y = np.mean(self.Y)
      self.SST = sum((self.Y - media_y) ** 2)
      self.SSE = sum(self.e_hat ** 2) 
      self.SSR = sum((self.y_hat - media_y)** 2)
      self.R2 = self.SSR / self.SST
    else:
      self.R2 = 1 - (np.sum(self.e_hat ** 2) / np.sum(self.Y ** 2))
    return self.R2


  def coef_ajt(self):
    """
    Calculando o coeficiente de determinação ajustado que serve para fazer comparações entre modelos
    """
    self.R2_ajt = 1 - (self.SSE / (self.T - self.K)) / (self.SST / (self.T - 1))
    return self.R2_ajt

  

