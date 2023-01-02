import numpy as np
import pandas as pd

#diferença
def dif(serie, t, n=1):
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
def defa(serie, t, n=1):
  if t == 0:
    return np.NaN
  else:
    return serie[t-n]

#retorno liquido
def ret_liq(serie, t):
  return(dif(serie, t) / defa(serie, t))


#retorno bruto
def ret_bruto(serie, t):
  return (serie[t]/defa(serie,t))

#criando uma nova série usando alguma das funções acima
def nova_serie(serie, func):
  copia = serie.copy()
  for t in range(serie.shape[0]):
    copia[t] = func(serie, t)
  return copia





