import numpy as np
import scipy.stats as sts
import pandas as pd



###1 Testes de Aderência pelo χ^2 - Pearson 1900


def estimador(categorias, freqs, distri, parametros_dados = []):
  """
  Função que estima os parâmetros das distribuições para o teste de chi quadradado
  (caso discreto)

  categorias: vetor numérico
  caterigas dos dados que vamos testar as distribuições

  freqs: vetor numérico
  frequências das respectivas categorias

  distribuicao: distribuição do modulo scipy.stats
  distribuição dos dados

  parametros_dados: vetor numérico
  parâmetros  já calculados usados na estimação dos demais parâmetros

  """

  parametros = np.array([])
  n = sum(freqs)


  if distri == 'binomial':
    size = parametros_dados[0]
    soma = 0
    for i in range(len(categorias)):
      soma += categorias[i] * freqs[i]
    p_hat = soma / (n*size)
    parametros = np.append(parametros, p_hat)

  elif distri == 'bernouli':
    soma = 0
    for i in range(len(categorias)):
      soma += categorias[i] * freqs[i]
    p_hat = soma / (n)
    parametros = np.append(parametros, p_hat)

  elif distri == 'poisson':
    soma += categorias[i] * freqs[i]
    lambda_hat = soma / n
    parametros = np.append(parametros, lambda_hat)
  
  return parametros



def test_chi_pearson(categorias, freqs, distribuicao, parametros_dados=[], estimar=''):
  """
  Testes chi quadrado de Aderência (caso discreto)
  Ho: todos valores de x tem a mesma distribuição
  H1: pelo menos 1 x tem uma distribuição diferente

  categorias: vetor numérico
  caterigas dos dados que vamos testar as distribuições

  freqs: vetor numérico
  frequências das respectivas categorias

  distribuicao: distribuição do modulo scipy.stats
  distribuição dos dados (se os parâmetros da distribuição não estiverem
  definidos o parâmetro estimar deve ser definido como o nome da distribuição)

  parametros_dados: vetor numérico
  parâmetros  já calculados usados na estimação dos demais parâmetros

  estimar: string:
  nome da distribuição da qual será estimada os parâmetros
  """

  n = np.sum(freqs)

  if estimar:
    if estimar == 'binomial':
      p_hat = estimador(categorias, freqs, estimar, parametros_dados)
      distribuicao = distribuicao(parametros_dados[0], p_hat)
    elif estimar == 'bernouli':
      p_hat = estimador(categorias, freqs, estimar, parametros_dados)
      distribuicao = distribuicao(p_hat)
    elif estimar == 'poisson':
      lambda_hat = estimador(categorias, freqs, estimar)
      distribuicao = distribuicao(lambda_hat)

    

  freqs_esperadas = n * distribuicao.pmf(categorias)
  quadrado_diferencas = (freqs - freqs_esperadas)**2

  T = np.sum((quadrado_diferencas)/freqs_esperadas)
  
  df = pd.DataFrame({'categorias':categorias, 'freqs':freqs, 'freqs_esperadas':freqs_esperadas, 'quadrado_diferencas':quadrado_diferencas})
  print(df)
  return T



def estimador_intervalo(x, freqs, distri, intervalos=[]):
  """
  Função que estima os parâmetros das distribuições para o teste de chi quadradado
  (caso contínuo)

  x: vetor numérico
  representante de um intervalo
  
  freqs: vetor numérico
  frequências dos intervalos

  intervalos: vetor de vetores
  vetor onde cada elemento é  uma lista
  composta do intervalo inferior e do superior

  distribuicao: distribuição do modulo scipy.stats
  distribuição dos dados 

  estimar: string:
  nome da distribuição da qual será estimada os parâmetros
  """
  xf = x*freqs
  n = np.sum(freqs)
  parametros = np.array([])
  
  if distri == 'normal':
    media = np.sum(xf)/n
    var = (np.sum((x**2)*freqs) - n * media**2) / (n - 1) 
    parametros = np.append(parametros, media)
    parametros = np.append(parametros, var)
  
  elif distri == 'exponencial':
    media = np.sum(xf)/n
    lambdaa = 1 / media
    parametros = np.append(parametros, lambdaa)

  elif distri == 'uniforme':
    if np.isinf(intervalos[0,0]):
      a = np.min(x)
    else:
      a = intervalos[0,0]
    if np.isinf(intervalos[intervalos.shape[0]-1,1]):
      b = np.max(x)
    else:
      b = intervalos[intervalos.shape[0]-1,1]
    parametros = np.append(parametros, a)
    parametros = np.append(parametros, b)
    
  return parametros


def test_chi_pearson_intevalos(intervalos, freqs, distribuicao, estimar=''):
  """
  Testes chi quadrado de Aderência (caso contínuo)
  Ho: todos valores de x tem a mesma distribuição
  H1: pelo menos 1 x tem uma distribuição diferente

  intervalos: vetor de vetores
  vetor onde cada elemento é  uma lista
  composta do intervalo inferior e do superior

  freqs: vetor numérico
  frequências dos respectivos intervalos

  distribuicao: distribuição do modulo scipy.stats
  distribuição dos dados (se os parâmetros da distribuição não estiverem
  definidos o parâmetro estimar deve ser definido como o nome da distribuição)

  estimar: string:
  nome da distribuição da qual será estimada os parâmetros
  """

  
  #calculando os valores de x como sendo o ponto médio dos limites dos intervalos
  amplitude = intervalos[1,1] - intervalos[1,0]
  x = np.array([])
  for i in range(intervalos.shape[0]):
    if i==0 and np.isinf(intervalos[i,0]):
      ponto_medio = (intervalos[i,1] + (intervalos[i,1] - amplitude)) / 2
      
    elif i== (intervalos.shape[0]-1) and np.isinf(intervalos[i,1]):
      ponto_medio = (intervalos[i,0] + (intervalos[i,0] + amplitude)) / 2
    else:
      ponto_medio = (intervalos[i,0] +intervalos[i,1]) / 2
    x = np.append(x, ponto_medio)
    

  #calulando as distribuições
  if estimar:
    if  estimar == 'normal':
      media_hat, var_hat = estimador_intervalo(x, freqs, distri=estimar)
      distribuicao = distribuicao(media_hat, var_hat**0.5)
    elif estimar == 'exponencial':
      lambdaa = estimador_intervalo(x, freqs, estimar)
      distribuicao = distribuicao(lambdaa)

  #calculando a padronização dos limites superiores dos intervalos
  z = np.array([])
  for i in range(len(x)):
    if (i == (intervalos.shape[0]-1)) and np.isinf(intervalos[i,1]):
      z = np.append(z,np.nan)
    else:
      ls = intervalos[i,1]
      z = np.append(z,ls)

  #calculando as acumuladas do z
  acumulada = np.array([])
  for i in range(len(z)):
    if np.isnan(z[i]):
      acumulada = np.append(acumulada, 1)
    else:
      acumulada = np.append(acumulada, distribuicao.cdf(z[i]))

  #calculando a p como das acumuladas
  p = np.array([])
  for i in range(len(acumulada)):
    if i==0:
      p = np.append(p, acumulada[i])
    else:
      p = np.append(p, acumulada[i] - acumulada[i-1]) 

  #calculando a estatística de teste
  freqs_esperadas = n * p
  quadrado_diferencas = (freqs - freqs_esperadas)**2
  quadrado_divido = (quadrado_diferencas)/freqs_esperadas
  T = np.sum(quadrado_divido)
  
  df = pd.DataFrame({'intervalos[0]':intervalos[:, 0], 'intervalos[1]':intervalos[:, 1], 'x':x, 'z':z, 'acumulada':acumulada,  'p':p,
                     'freqs':freqs, 'freqs_esperadas':freqs_esperadas, 'quadrado_diferencas':quadrado_diferencas,
                     'quadrado_divido':quadrado_divido})
  print(df)
  return T 


#2.3.2 Teste de Lilliefors

def test_lilli(dados):

  media = np.mean(dados)
  var = np.var(dados, ddof=1) #ddof é pra que a variancia fique sobre N-ddof

  dados_ordenados = np.sort(dados)
  prob_empirica = (np.argsort(dados_ordenados)+1)/len(dados_ordenados)

  distribuicao = sts.norm(loc=media, scale=var)
  prob = distribuicao.cdf(dados_ordenados)

  dados_ordenados_padronizados = (dados_ordenados - media)/var**(0.5)
  prob_padronizada = sts.norm().cdf(dados_ordenados_padronizados)

  diferencas = np.abs(prob_padronizada - prob_empirica)
  T = np.max(diferencas)

  df = pd.DataFrame({'dados_ordenados':dados_ordenados, 'prob_empirica':prob_empirica,
                     'prob':prob, 'prob_padronizada':prob_padronizada, 'diferencas':diferencas})
  print(df)

  return(T)


#3 Teste de Kolmogorov- Sminorv

def test_Kolmogorov_Sminorv(dados, distribuicao):

  media = np.mean(dados)
  var = np.var(dados, ddof=1) #ddof é pra que a variancia fique sobre N-ddof

  dados_ordenados = np.sort(dados)
  prob_empirica = (np.argsort(dados_ordenados)+1)/len(dados_ordenados)

  prob = distribuicao.cdf(dados_ordenados)


  diferencas = np.abs(prob - prob_empirica)
  T = np.max(diferencas)

  df = pd.DataFrame({'dados_ordenados':dados_ordenados, 'prob_empirica':prob_empirica,
                     'prob':prob, 'diferencas':diferencas})
  print(df)

  return(T)




#2.3.3 Teste de Jarque-Bera

def mk(dados, k):
  media = np.mean(dados)
  n = len(dados)
  m = np.sum((dados - media)**k) / n
  return(m)


def test_Jarque_Bera(dados):
  n = len(dados)
  var = np.var(dados, ddof=1)
  assimetria = mk(dados, 3) / (var**(3/2))
  curtose = mk(dados, 4) / (var**(2))
  T = n * ( (assimetria/6) + ((curtose+3)**2)/24)
  return(T)



#Teste do sinal
def test_sinal(x, y):
  sinais = []
  positivo = 0
  negativo = 0
  zero = 0
  for i in range(x.shape[0]):
    if y[i] > x[i]:
      sinais.append(1)
      positivo += 1
    elif y[i] < x[i]:
      sinais.append(-1)
      negativo +=1
    else:
      sinais.append(0)
      zero += 1
  sinais = np.array(sinais)
  df = pd.DataFrame({'x':x, 'y':y, 'sinais':sinais})
  print(df)
  if positivo == negativo:
    T = positivo + (zeros/2)
  else:
    T = positivo
  return T




def sorteio(vetor):  
  #eu tive que criar essa função pq o np.arsort não estava funcionando direito
  #(ou como eu acho que devia funcionar)
  lista = vetor.tolist()
  ordenado = sorted(lista)
  indices = []
  
  contadores = {}
  for c in lista:
    contadores[c] = 0

  for i in range(len(lista)):
    indices.append(ordenado.index(lista[i]) + contadores[lista[i]])
    contadores[lista[i]] += 1
  return np.array(indices)


def ajeita_posto(vetor):
  #criando o vetor e os postos dos valores > 0
  vetor_sem_0 = vetor[vetor > 0].copy() # !=0 tbm serve
  posto_sem_0 = sorteio(vetor_sem_0) + 1 #o + 1 é pq o primeiro indice é o 1
  posto_sem_0 = np.array(posto_sem_0, dtype='float32')
  
  posto_ver = posto_sem_0.copy()#criando essa variável extra pra ver como ela fica sem mudar os valores repetidos

  #reatribuindo os valores ao posto_sem_0 para encaixar com os valores repetidos
  for i in range(len(posto_sem_0)):
    indice_repetidos =vetor_sem_0 == vetor_sem_0[i]
    postos_repetidos = posto_sem_0[indice_repetidos].copy()
    if len(postos_repetidos) > 1:
      soma_posto_repetidos = np.sum(postos_repetidos)
      quant_repetidos = len(postos_repetidos)
      posto_sem_0[indice_repetidos] =  soma_posto_repetidos / quant_repetidos
      posto_ver[i] = posto_ver[i]
  
  #criando a lista final dos postos mas com os postos dos 0's como np.nan pra encaixar o tamanho
  posto = []
  posto_ver2 = []
  cont = 0
  for i in range(len(vetor)):
    if vetor[i] == vetor_sem_0[cont]:
      posto.append(posto_sem_0[cont])
      posto_ver2.append(posto_ver[cont])
      cont +=1
    else:
      posto.append(np.nan)
      posto_ver2.append(np.nan)
  return [np.array(posto), posto_ver2]


def teste_wilcoxon(x, y):
  #O tempo e estresse que eu passei criando esse teste e as duas funções acima não tá escrito nessas linhas de código
  
  d = y - x
  abs_d = np.abs(d)
  posto, posto_ver = ajeita_posto(abs_d)
  posto_sinal = []
  for i in range(posto.shape[0]):
    if d[i] < 0:
      posto_sinal.append(-posto[i])
    elif d[i] > 0:
      posto_sinal.append(posto[i])
    else:
      posto_sinal.append(np.nan)
  posto_sinal = np.array(posto_sinal)
  df = pd.DataFrame({'x':x, 'y':y, 'd':d, 'abs_d':abs_d, 'posto_ver':posto_ver, 'posto':posto, 'posto_sinal':posto_sinal})
  print(df.sort_values(by='abs_d'))
  V = 0
  for c in posto_sinal:
    if c > 0:
      V += c
  return(V)




