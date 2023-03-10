import numpy as np
import pandas as pd

#diferença
def diferenca(serie, t, n=1):
  """Função que realiza a operação de diferença num elemento numa série temporal
  
  serie: vetor numérico
  série temporal da qual será calculada a diferença

  t: número inteiro
  valor do indice do qual sera calculada a diferença

  n: número inteiro
  potência da diferença
  """

  if t == 0:
    return serie[t]
  else:
    if n == 1:
      return (serie[t] - serie[t-1])
    else:
      soma = serie[t]
      for i in range(n):
        soma += (-1)**(i+1) * (n - i) * serie[t-i-1]
      return(soma)


#defasagem
def defasagem(serie, t, n=1):
  """Função que realiza a operação de defasagem num elemento numa série temporal
  
  serie: vetor numérico
  série temporal da qual será calculada a defasagem

  t: número inteiro
  valor do indice do qual sera calculada a defasagem

  n: número inteiro
  potência da defasagem
  """

  if t == 0:
    return np.NaN
  else:
    return serie[t-n]

#retorno liquido
def retorno_liquido(serie, t):
  """Função que realiza a operação de retorno liquido num elemento numa série temporal
  
  serie: vetor numérico
  série temporal da qual será calculado o retorno liquido

  t: número inteiro
  valor do indice do qual sera calculada o retorno liquido
  """
  return(diferenca(serie, t) / defasagem(serie, t))

#retorno bruto
def retorno_bruto(serie, t):
  """Função que realiza a operação de retorno bruto num elemento numa série temporal
  
  serie: vetor numérico
  série temporal da qual será calculado o retorno bruto

  t: número inteiro
  valor do indice do qual sera calculada o retorno bruto
  """
  return (serie[t]/defasagem(serie,t))

#criando uma nova série usando alguma das funções acima
def nova_serie(serie, func):
  """Cria uma nova série baseada numa das funções anteriores (na prática pode ser qualque função)

  serie: vetor numérico
  série temporal que será usada na criação da nova série

  func: função
  Função que será aplicada as elementos da série original
  """
  copia = serie.copy()
  for t in range(serie.shape[0]):
    copia[t] = func(serie, t)
  return copia





