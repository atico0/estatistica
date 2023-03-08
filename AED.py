def media(x):
  """função da média aritimética
  
  x vetor numérico
  vetor com os dados dos quais será calculad1 a média"""
  v = np.sum(x) / len(x)
  return v


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
  valor no qual o momento está centrado"""

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