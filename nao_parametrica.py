import numpy as np
import scipy.stats as sts
import pandas as pd



###1 Testes de Aderência pelo χ^2 - Pearson 1900


def test_chi_pearson(categorias, freqs, distribuicao):
  n = np.sum(freqs)

  freqs_esperadas = n * distribuicao.pmf(categorias)
  quadrado_diferencas = (freqs - freqs_esperadas)**2

  T = np.sum((quadrado_diferencas)/freqs_esperadas)
  
  df = pd.DataFrame({'categorias':categorias, 'freqs':freqs, 'freqs_esperadas':freqs_esperadas, 'quadrado_diferencas':quadrado_diferencas})
  print(df)
  return(T)


#esse vai ser o teste para intervalos de valores
def test_chi_pearson_intevalos(intervalos, freqs, distribuicao):
  n = np.sum(freqs)
  p = []
  for i in range(intervalos.shape[0]):
    p.append(distribuicao.cdf(intervalos[i][1]) - distribuicao.cdf(intervalos[i][0]))
  p = np.array(p)
  freqs_esperadas = n * p
  quadrado_diferencas = (freqs - freqs_esperadas)**2

  T = np.sum((quadrado_diferencas)/freqs_esperadas)
  df = pd.DataFrame({'intervalos':intervalos, 'freqs':freqs, 'p':p, 'freqs_esperadas':freqs_esperadas, 'quadrado_diferencas':quadrado_diferencas})
  print(df)
  return(T)


#esse vai ser o teste para intervalos de valores (padronizados)
def test_chi_pearson_intevalos_padronizados(intervalos, freqs, distribuicao):
  n = np.sum(freqs)
  p = []
  for i in range(intervalos.shape[0]):
    p.append(distribuicao.cdf(intervalos[i][1]) - distribuicao.cdf(intervalos[i][0]))
  p = np.array(p)
  freqs_esperadas = n * p
  quadrado_diferencas = (freqs - freqs_esperadas)**2

  T = np.sum((quadrado_diferencas)/freqs_esperadas)
  df = pd.DataFrame({'intervalos':intervalos, 'freqs':freqs, 'p':p, 'freqs_esperadas':freqs_esperadas, 'quadrado_diferencas':quadrado_diferencas})
  print(df)
  return(T)



#2- Teste de Lilliefors 



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
  #eu tive que criar essa função pq o np.arsort não estava funcionando direito (ou como eu acho que devia funcionar)
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
  # MDS o tempo e estresse que eu passei criando esse teste e as duas funções acima não tá escrito nessas linhas de código

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




