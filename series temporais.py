import numpy as np
import pandas as pd
import scipy.stats as sts
import AED



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



def filtragem(serie, t, q, s, a):
  """
  função de filtragem de um elemento numa série temporal

  serie: vetor numérico
  série da qual será feita a filtragem

  t: número inteiro
  indíce do elemento da serie

  q: número inteiro
  s: número inteiro
  início e fim dos elemtos da filtragem

  a: vetor ou escalar numérico
  vetor de pesos
  """

  #obs: como o indice no python começam do 0  mas as series começam de T=1
  #isso faz parecer que esse linha tá errada e que devia ser t - q < 1
  if t - q < 0:
    print("t - q precisa ser maior que 0")
    return np.NaN
  if t + s > len(serie):
    print("q + s precisa ser menor que o tamanho da série")
    return np.NaN

  if type(a) == 'int' or type(a) =='float':
    a = np.ones(tamanho) * a
    tamanho = q + s + 1
  nova_serie = np.array([])
  soma = 0
  for k in range(-q, s+1):
    soma += a[k] * serie[t+k]
  return soma



def autocov(serie, k=1):
  """
  Função da autocovariância estimada

  serie: vetor numérico
  Série da qual será calculada a estimativa da autocovariância

  k: valor inteiro
  'lag' da autocovariância
  """
  T = len(serie)
  media = AED.media(serie)
  media_k = AED.momentos(serie, 1, k=k, c=0)
  soma = 0
  for t in range(0, T - k):
    soma += (serie[t] - media) * (serie[t + k] - media_k)
  return soma




def autocorr(serie, k):
  """
  Função da autocorrelação estimada

  serie: vetor numérico
  Série da qual será calculada a estimativa da autocovariância

  k: valor inteiro
  'lag' da autocovariância
  """
  result = autocov(serie=serie, k=k) / autocov(serie=serie, k=0)
  return result




def cria_AR(t, phi, mu=0, p=1, distribuicao=sts.norm()):
  """
  criando série temporal pelo processo auto regressivo de ordem q (AR(p))

  t: valor inteiro
  tamanho da série

  phi: vetor numérico
  conjunto dos coeficientes das observações anteriores

  mu: valor numérico
  constante da série

  p: valor inteiro
  ordem da série

  distribuicao: objeto do tipo scipy.stats
  distribuição de probabilidade dos ruídos brancos
  """
  #vetor dos ruidos brancos
  e = distribuicao.rvs(t)
  #criando o y
  y = np.array([])
  for i in range(t):
    soma = 0
    for j in range(p):
      if i - j > 0 and j > 0:
        soma += y[i - j] * phi[j]
      
    novo_termo = soma + mu + e[i]
    y = np.append(y, novo_termo)
  return y


      
def cria_MA(t, theta, mu=0, q=1, distribuicao=sts.norm()):
  """
  criando série temporal pelo processo de médias moveis de ordem q (MA(q))

  t: valor inteiro
  tamanho da série

  theta: vetor numérico
  conjunto dos coeficientes dos ruídos anteriores

  mu: valor numérico
  constante da série

  q: valor inteiro
  ordem da série

  distribuicao: objeto do tipo scipy.stats
  distribuição de probabilidade dos ruídos brancos
  """
  #vetor dos ruidos brancos
  e = distribuicao.rvs(t)
  #criando o y
  y = np.array([])
  for i in range(t):
    soma = 0
    for j in range(q):
      if i - j > 0 and j > 0:
          soma += e[i - j] * theta[j]
    
    novo_termo = soma + mu + e[i]
    y = np.append(y, novo_termo)
  return y




def cria_ARMA(t, theta, phi, mu=0, q=1, p=1, distribuicao=sts.norm()):
  """
  criando série temporal pelo processo auto regressivo de médias móveis de ordem q (ARMA(p,q))

  t: valor inteiro
  tamanho da série

  theta: vetor numérico
  conjunto dos coeficientes dos ruídos anteriores

  phi: vetor numérico
  conjunto dos coeficientes das observações anteriores

  mu: valor numérico
  constante da série

  p: valor inteiro
  q: valor inteiro
  ordens da série

  distribuicao: objeto do tipo scipy.stats
  distribuição de probabilidade dos ruídos brancos
  """
  #vetor dos ruidos brancos
  e = distribuicao.rvs(t)
  #criando o y
  y = np.array([])
  for i in range(t):
    soma_theta = 0
    soma_phi = 0

    for j in range(q):
      if i - j > 0 and j > 0:
        soma_theta += e[i - j] * theta[j]
    
    for j in range(p):
      if i - j > 0 and j > 0:
        soma_phi += y[i - j] * theta[j]
      
    novo_termo = soma_theta + soma_phi + mu + e[i]
    y = np.append(y, novo_termo)
  return y