import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math




def HT(amostra, va, prob):
  soma = []
  for i in range(len(amostra)):
    if (not np.isnan(amostra[va].iloc[i])) and not np.isnan(amostra[prob].iloc[i]):
      soma.append(((amostra[va].iloc[i]) / (amostra[prob].iloc[i])))
  return sum(soma)



def erros_relativos(reais, previstos, media=False):
  lista = []
  for c in range(len(reais)):
    lista.append(abs( (reais[c] - previstos[c]) ) / reais[c])
  if media:
    return (lista, np.mean(lista))
  return lista




'''
1) É definida uma  série u de n realizações independentes de uma Uniforme(0,1) (u1, u2, ..., uN)

2) se u1 < (N/n), então k = 1 pertence a amostra S senão k = 1 não pertence a amostra S

3) para k = 2, 3, ...N se  uk < ( (n - nk) / (N - k + 1) ), então k pertence a amostra S senão k não pertence a amostra S 
onde  nk é o número de elementos selecionados para compor a amostra dentre os primeiros k - 1 elementos listados no cadastro.

4) O processo continua até que nk = n
'''

def AAS(df, n=0, proporcao=0):
  if n==0:
    n = int(len(df)*proporcao)
  N = df.shape[0]
  amostra = []
  uniforme = np.random.uniform(low=0, high=1, size=N)
  prob = n/N
  if uniforme[0]<prob:
    amostra.append(0)
  i=2
  while len(amostra) != n:
    if  uniforme[i-1] < (n-len(amostra))/(N-i+1):
      amostra.append(i-1)
    i += 1
  novo_df = df.iloc[amostra,:].copy()
  novo_df['unifome'] = uniforme[amostra].copy()
  novo_df['prob'] = prob
  return novo_df


def HT_AAS(amostra, va, N):
  return (amostra[va].mean())*N


def V_AAS(amostra, df, va):
  N = len(df)
  n = len(amostra)
  S2y = df[va].var()
  return (N**2) * ( (1/n) - (1/N) ) * S2y


def Vh_AAS(amostra, df, va):
  N = len(df)
  n = len(amostra)
  S2hy = (amostra[va].var()) * n / (n - 1)
  return (N**2) * ( (1/n) - (1/N) ) * S2hy



#def AES(df, estratos, n=0, proporcao=0):
#  if n==0:
#    n = int(len(df)*proporcao)
#  novo_df = pd.DataFrame(columns=df.columns)
#  estratos_vals = np.unique(df[estratos])
 # for i in range(len(estratos_vals)):
#    estrato = df[df[estratos]==estratos_vals[i]].copy()
#    novo_df = pd.concat((novo_df, AAS(estrato, int(n*(len(estrato)/len(df)) ) )), axis=0)
#  return novo_df



   
'''
1) É definida uma constante 0 < pi < 1
2) É definida uma u série de n realizações independentes de uma Uniforme(0,1) (u1, u2, ..., uN)
se uk < pi, então k pertence a amostra S
senão k não pertence a amostra S
'''
def AB(df, prob):
  uniforme = np.random.uniform(0,1,df.shape[0])
  amostra = []
  for i in range(df.shape[0]):
    if uniforme[i]<=prob:
      amostra.append(i)
  novo_df = df.iloc[amostra].copy()
  novo_df['unifome'] = uniforme[amostra]
  novo_df['prob'] = prob
  return novo_df


def HT_AB(amostra, va, prob):
  soma = amostra[va].sum()
  return soma / prob


def V_AB(amostra, df, va, prob):
  soma = 0
  for i in range(len(df)):
    if (not np.isnan(amostra[va].iloc[i])):
      soma += (df[va].iloc[i])**2
  return ( (1/prob) - 1)*soma


def Vh_AB(amostra, va, prob):
  soma = 0
  for i in range(len(amostra)):
    if (not np.isnan(amostra[va].iloc[i])):
      soma += (amostra[va].iloc[i])**2
  return (1/prob) * ( (1/prob) - 1) * soma





#def AEB(df, estratos, probs):
#  novo_df = pd.DataFrame(columns=df.columns)
 # estratos_vals = np.unique(df[estratos])
#  for i in range(len(estratos_vals)):
#    estrato = df[#df[estratos]==estratos_vals[i]].copy()
#    novo_df = pd.concat((novo_df, AB(estrato, probs[i])), axis=0)
#  return novo_df






def POI(df, vetor_prob):
  uniforme = np.random.uniform(0,1,df.shape[0])
  amostra = []
  for i in range(df.shape[0]):
    if uniforme[i]<=vetor_prob[i]:
      amostra.append(i)
  novo_df = df.iloc[amostra].copy()
  novo_df['unifome'] = uniforme[amostra]
  return novo_df

'''
1) É definida  uma série  pi = (pi1, ..., piN) onde todos elementos de pi são menores que 0 e maiores que 1
2) É definida uma série u de n realizações independentes de uma Uniforme(0,1) (u1, u2, ..., uN )
3) Se uk < pik, então k pertence a amostra S, senão k não pertence a amostra S
'''


def AEPOI(df, estratos, probs):
  novo_df = pd.DataFrame(columns=df.columns)
  estratos_vals = np.unique(df[estratos])
  for i in range(len(estratos_vals)):
    estrato = df[df[estratos]==estratos_vals[i]].copy()
    novo_df = pd.concat((novo_df, POI(estrato, probs[i])), axis=0)
  return novo_df




def PPM(df, probs, dropar=False):
  uniforme = np.random.uniform(0,1,df.shape[0])
  amostra = []
  for i in range(df.shape[0]):
    if uniforme[i] < probs[i]:
      amostra.append(i)
  novo_df = df.iloc[amostra].copy()
  novo_df['unifome'] = uniforme[amostra]
  return novo_df





'''
1) É definida uma variável auxiliar x com X = sum(x)
2) É gerado uma série u  de k variáveis aleatórias que seguem uma distribuição uniforme(0,1)
3) É calculado pik = xk / X onde xk é o k-ésimo elemento da coluna x
4) É calculado lambdak = n*pk
5) É calculado Ak = uk/lambdak onde uk é o k-ésimo elemento do vetor u
6) São selecionados para a amostra as n unidades com os menores valores de Ak
'''

def ASP(df, auxiliar, n=0, proporcao=0, manter = False):
  if n==0:
    n = int(len(df)*proporcao)
  soma = df[auxiliar].sum()
  df['uniforme'] = np.random.uniform(low=0, high=1, size=len(df))
  df['pi'] = [ (df[auxiliar].iloc[i])/soma for i in range(len(df))]
  df['modificado'] = np.nan
  for i in range(n):
    df['modificado'].iloc[i] = (df['uniforme'].iloc[i])/ (df['pi'].iloc[i])
  novo_df = df.sort_values(by='modificado', ascending=True).iloc[:n, :].copy()
  return novo_df



def HT_ASP(amostra, va, pi):
  soma = 0
  for i in range(len(amostra)):
    if (not np.isnan(amostra[va].iloc[i])) and not np.isnan(amostra[pi].iloc[i]):
      soma += (amostra[va].iloc[i] / amostra[pi].iloc[i])
  return soma / len(amostra)



def V_ASP(amostra, df, va, pi):
  N = len(df)
  n = len(amostra)
  total_va = df[va].sum()
  soma = 0
  for i in range(N):
    p_i = df[pi].iloc[i]
    y_i = df[va].iloc[i]
    if (not np.isnan(y_i)) and not np.isnan(p_i):
      soma += (1 - n * p_i) * p_i * ((y_i / p_i) - total_va)**2
  return (N/(N-1))*(1/n)*soma


def Vh_ASP(amostra, va, pi):
  n = len(amostra)
  estimativa_va = HT_ASP(amostra, va, pi)
  soma = 0
  for i in range(n):
    p_i = amostra[pi].iloc[i]
    y_i = amostra[va].iloc[i]
    if (not np.isnan(y_i)) and not np.isnan(p_i):
      soma += ( ((y_i / p_i) - estimativa_va)**2) * (1-n*p_i) * p_i
  return (1/(n*(n-1))) * soma



#def AESP(df, estratos, auxiliares, n=0, proporcao=0):
#  if n==0:
#    n = int(len(df)*proporcao)
#  novo_df = pd.DataFrame(columns=df.columns)
#  estratos_vals = np.unique(df[estratos])
#  for i in range(len(estratos_vals)):
#    df[df[estratos]==estratos_vals[i]]
#  return novo_df







#3)



#OBS: ERRO PADRÃO É O DESVIO PADRÃO DE UM ESTIMADOR EP(T) (COMO EU JÁ DEFINI AS VARS ENTRE OS ESTIMADORES É SÓ TIRAR A RIZ)
# COEFICIENTE DE VARIAÇÃO = (EP(T) / MÉDIA AMOSTRAL )* 100