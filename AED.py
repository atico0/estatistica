import numpy as np
import pandas as pd
import utils

def media(x):
  """função da média aritimética
  
  x vetor numérico
  vetor com os dados dos quais será calculada a média"""
  v = np.sum(x) / len(x)
  return v

def moda(x):
  """
  função que calcula media modal (o elemento que mais aparece)
  obs: caso varios elementos tenham a maior frequência,
  a função retornará uma lista com as modas

  x vetor numérico
  vetor com os dados dos quais será calculada a moda
  """
  modas = []
  dicio = utils.contar_unicos(x)
  maior_contagem = max(list(dicio.keys()))
  for i in range(len(x)):
    if dicio[x[i]] == maior_contagem:
      modas.append(x[i])
  if len(modas) == 1:
    return modas[0]  
  return modas

def mediana(lista):
  """
  função que calcula media mediana (o elemento do meio dos dados ordenados)

  x vetor numérico
  vetor com os dados dos quais será calculada a mediana
  """

  lista_ordenada = nao_parametrica.ordena(lista)
  tamanho = len(lista_ordenada)
  if tamanho % 2 == 0:
      mediana = (lista_ordenada[int(tamanho/2) - 1] + lista_ordenada[int(tamanho/2)]) / 2
  else:
      mediana = lista_ordenada[int(tamanho/2)]
  return mediana


def media_geo(x):
  """função da média geométrica

  x vetor numérico
  vetor com os dados dos quais será calculada a média"""

  n = len(x)
  prod = 1
  for i in range(len(x)):
    prod = prod * x[i]
  prod = prod **(1/n)
  return prod


def media_ponderada(x, p):
  """função da média ponderada
  x vetor  numérico
  vetor com os dados dos quais será calculada a média

  p vetor de inteiros
  pesos que serão atribuidos ao vetor x no cálculo da média"""

  n = sum(p)
  soma = 0
  for i in range(len(x)):
    soma += x[i] * p[i]
  media = soma / n
  return media



def momentos(x, momento, k=0, c='média'):
  """função genérica de momentos

  x: vetor de valores numéricos
  vetor com os dados dos quais será calculado o momento

  momento: valor númerico
  ordem do momento calculado

  k: valor inteiro
  usado no calculo de funções como a variância amostral

  c: valor numérico
  valor no qual o momento está centrado
  """

  if c == 'média':
    c = media(x)
  n = x.shape[0]
  v = sum((x - c)**momento) / (n - k)
  return v


def var_pop(x):
  """Variância populacional
  
  x: vetor de parâmetros numéricos
  vetor com os dados dos quais será calculada a variância
  
  interpretação: Quanto maior a variância, mais os dados tendem a se dispersar da média"""

  result = momentos(x, 2, 0)
  return result


def var_am(x):
  """Variância amostral
  x: vetor numérico
  vetor com os dados dos quais será calculada a variância
  interpretação: Quanto maior a variância, mais os dados tendem a se dispersar da média"""

  result = momentos(x, 2, 1)
  return result


def desvio_padrao_pop(x):
  """Desvio padrão populacional
  x: vetor numérico
  interpretação: Quanto maior a o desvio padrão, mais os dados tendem a se dispersar da média"""

  result = (momentos(x, 2, 0))**0.5
  return result


def desvio_padrao_am(x):
  """Desvio padrão amostral
  x: vetor numérico
  interpretação: Quanto maior a o desvio padrão, mais os dados tendem a se dispersar da média"""

  result = (momentos(x, 2, 1))**0.5
  return result


def coef_assimetria_pop(x):
  """coeficiente de assimetria populacional
  x: vetor numérico
  vetor com os dados dos quais será calculada a assimetria
  
  interpretação:
  assimetria > 0 implica que os dados estão concentrados a esquerda (em valores baixos)
  assimetria < 0 implica que os dados estão concentrados a direita (em valores altos)
  assimetria = 0 implica que os dados estão concentrados a no centro (em valores altos)
  quanto maior o |assimetria| maior a concentração de dados"""

  result = momentos(x, 3, 0) / (desvio_padrao_pop(x)**3)
  return result


def coef_assimetria_am(x):
  """coeficiente de assimetria amostral (usando o desvio padrão)
  
  x: vetor numérico
  vetor com os dados dos quais será calculada a assimetria
  
  interpretação:
  assimetria > 0 implica que os dados estão concentrados a esquerda (em valores baixos)
  assimetria < 0 implica que os dados estão concentrados a direita (em valores altos)
  assimetria = 0 implica que os dados estão concentrados a no centro (em valores altos)
  quanto maior o |assimetria| maior a concentração de dados """

  n = len(x)
  mean  = media(x)
  result = n * np.sum(((x - mean) / desvio_padrao_am(x))**3) / ((n-1) * (n-2))
  return result


def coef_curtose_pop(x):
  """coeficiente de curtose populacional
  x: vetor numérico
  vetor com os dados dos quais será calculada a curtose
  
  interpretação:
  curtose > 3 implica que os dados são menos achatados (ou mais altos) que a normal (leptocúrtica) 
  curtose < 3 implica que os dados são mais achatados (ou menos altos) que a normal (platicúrtica)
  curtose = 3 implica que os dados tem o  mesmo achatamento que a normal. (mesocúrtica)
  quanto maior a curtose maior é a concentração de dados em torno do média e menor é a probabilidade de eles assumirem valores extremos"""
  
  result = momentos(x, 4, 0) / (desvio_padrao_pop(x)**4)
  return result
  

def coef_curtose_am(x):
  """coeficiente de curtose amostral (usando o desvio padrão)

  x: vetor numérico
  vetor com os dados dos quais será calculada a curtose
  
  interpretação:
  curtose > 3 implica que os dados são menos achatados (ou mais altos) que a normal (leptocúrtica) 
  curtose < 3 implica que os dados são mais achatados (ou menos altos) que a normal (platicúrtica)
  curtose = 3 implica que os dados tem o  mesmo achatamento que a normal. (mesocúrtica)
  quanto maior a curtose maior é a concentração de dados em torno do média e menor é a probabilidade de eles assumirem valores extremos"""
  
  mean  = media(x)
  n = len(x)
  result = n * (n+1) * np.sum(((x - mean) / desvio_padrao_am(x))**4) / ((n-1) * (n-2) * (n-3))
  return result

def covariancia_pop(x, y):
  """
  Função populacional da covariância

  x: vetor numérico
  y: vetor numérico
  variáveis das quais será calculada a variância

  interpretação:
  quanto maior o modulo da covariância maior será a interferncia de x em y
  (e vice versa). No caso da covariância positiva, um aumento em x implica num
  aumento em y e no caso da covariância negativa, um aumento em y
  implica um decaimento em y
  """
  media_x = media(x)
  media_y = media(y)
  cov = media((x - media_x) * (y - media_y))
  return cov


def correlacao_pop(x, y):
  """
  Função populacional da correlação

  x: vetor numérico
  y: vetor numérico
  variáveis das quais será calculada a variância

  interpretação:
  quanto maior o modulo da correlação maior será a interferncia de x em y
  (e vice versa). No caso da correlação positiva, um aumento em x implica num
  aumento em y e no caso da correlação negativa, um aumento em y
  implica um decaimento em y
  
  |corr| = 0 => correlação inexistente
  0 < |corr| <= 0.19 => correlação muito fraca
  0.19 < |corr| <= 0.39 => correlação fraca
  0.39 < |corr| <= 0.69 => correlação moderada
  0.69 < |corr| <= 0.89 => correlação forte
  0.89 < |corr| < 1 => correlação muiot forte
  |corr| = 1 => correlação perfeita
  """
  cov = covariancia_pop(x,y)
  dp_x = desvio_padrao_pop(x)
  dp_y = desvio_padrao_pop(y)
  corr = cov / (dp_x * dp_y)
  return corr

def covariancia_am(x, y):
  """
  Função amostral da covariância

  x: vetor numérico
  y: vetor numérico
  variáveis das quais será calculada a variância

  interpretação:
  quanto maior o modulo da covariância maior será a interferncia de x em y
  (e vice versa). No caso da covariância positiva, um aumento em x implica num
  aumento em y e no caso da covariância negativa, um aumento em y
  implica um decaimento em y
  """
  media_x = media(x)
  media_y = media(y)
  n = len(x)
  cov = momentos((x - media_x) * (y - media_y), k=0)
  return cov


def correlacao_am(x, y):
  """
  Função amostral da correlação

  x: vetor numérico
  y: vetor numérico
  variáveis das quais será calculada a variância

  interpretação:
  quanto maior o modulo da correlação maior será a interferncia de x em y
  (e vice versa). No caso da correlação positiva, um aumento em x implica num
  aumento em y e no caso da correlação negativa, um aumento em y
  implica um decaimento em y
  
  |corr| = 0 => correlação inexistente
  0 < |corr| <= 0.19 => correlação muito fraca
  0.19 < |corr| <= 0.39 => correlação fraca
  0.39 < |corr| <= 0.69 => correlação moderada
  0.69 < |corr| <= 0.89 => correlação forte
  0.89 < |corr| < 1 => correlação muiot forte
  |corr| = 1 => correlação perfeita
  """
  cov = covariancia_am(x,y)
  dp_x = desvio_padrao_pop(x)
  dp_y = desvio_padrao_pop(y)
  corr = cov / (dp_x * dp_y)
  return corr