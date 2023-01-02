import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def padronizador(X):
  return StandardScaler().fit_transform(X)


def correlacao(Z):
  Z_linha = np.matrix.transpose(Z)
  corr = np.linalg.multi_dot([Z_linha, Z])
  if corr[0,0] == 1:
    return corr
  return(corr/corr[0,0])



def calcula_vifs(Z):
  inversa = np.linalg.inv(Z)
  return np.diag(inversa)


def autosistema(Z):
  Z_linha = np.matrix.transpose(Z)
  Z_linha_Z = np.linalg.multi_dot([Z_linha, Z])

  autovalores, autovetores = np.linalg.eig(Z_linha_Z)
  k = np.max(autovalores) /np.min(autovalores)
  indices = np.max(autovalores)/ autovalores
  indice_k = np.argmin(autovalores)
  autovetor_k = autovetores[k, :]
  produto = np.linalg.multi_dot([autovetor_k, Z])
  return [autovetores, autovalores, k, indices, indice_k, autovetor_k, produto]


def determinante(Z):
  Z_linha = np.matrix.transpose(Z)
  produto = np.linalg.multi_dot([Z_linha, Z])
  det = np.linalg.det(produto)
  return(det)


def diagnostico(X):
  Z = padronizador(X)
  print('matriz padronizada:')
  print(Z)
  print('=-'*30)
  print('=-'*30)

  corr = correlacao(Z)
  print("matriz de correlaçÕES de X:")
  print(corr) 
  print('Quanto mais proximos de 1 ou -1 os elementos estiverem, maior a correlação entre suas variáves')
  print('=-'*30)
  print('=-'*30)

  vifs = calcula_vifs(Z)
  print('Vifs:')
  print(vifs)
  print("Os vifs corresponde a diagonal de (Z'Z)^(-1)")
  print('um ou mais vifs de valor alto (maior que 5 ou que 10) indicam a existência de multicolinearidade')
  print('Também é uma indicação de que os coeficientes de regressão associados estão sendo mal estimados por causa da multicolinearidade')
  print('=-'*30)
  print('=-'*30)

  
  autovetores, autovalores, k, indices, indice_k, autovetor_k, produto = autosistema(Z)
  print("autovetores de Z'Z")
  print(autovetores)
  print("autovalores de Z'Z")
  print(autovalores)
  print("número de condição de Z'Z (k):")
  print(k)
  print('k < 100 => não existe um problema sério de multicolinearidade')
  print('100 < k < 1000 => existe um problema moderado ou sérido de multicolinearidaed')
  print('k > 1000 => multicolinearidade severa')
  print('=-'*30)

  print("indices de condição:")
  print(indices)
  print('O número de indices de condição com valores grandes pode ser usado para medir a quantidade de dependencias lineares na matriz')
  print("O número de condição é o indice de condição na posição: ", indice_k)
  print('=-'*30)
  
  print('O autovetor com indice correspondente ao indice do número de condição é o:')
  print(autovetor_k)
  print('teoricamente, o produto entre esse autovetor e X chega mais perto do vetor nulo que o produto entre qualquer outro vetor e X ')
  print('produto:')
  print(produto)
  print('=-'*30)
  print('=-'*30)

  det = determinante(Z)
  print("|Z'Z|:")
  print(det)
  print("quanto mais proximo esse determinante for de 0, maior o grau de multicolinearidade")
