import numpy as np
import pandas as pd
import scipy.stats as sts
import AED
import regressao1


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


# retorno liquido
def retorno_liquido(serie, t):
    """Função que realiza a operação de retorno liquido num elemento numa série temporal

    serie: vetor numérico
    série temporal da qual será calculado o retorno liquido

    t: número inteiro
    valor do indice do qual sera calculada o retorno liquido
    """
    return(diferenca(serie, t) / defasagem(serie, t))


# retorno bruto
def retorno_bruto(serie, t):
    """Função que realiza a operação de retorno bruto num elemento numa série temporal

    serie: vetor numérico
    série temporal da qual será calculado o retorno bruto

    t: número inteiro
    valor do indice do qual sera calculada o retorno bruto
    """
    return (serie[t]/defasagem(serie, t))


# criando uma nova série usando alguma das funções acima
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

    # obs: como o indice no python começam do 0  mas as series começam de T=1
    # isso faz parecer que esse linha tá errada e que devia ser t - q < 1
    if t - q < 0:
        print("t - q precisa ser maior que 0")
        return np.NaN
    if t + s > len(serie):
        print("q + s precisa ser menor que o tamanho da série")
        return np.NaN

    if type(a) == 'int' or type(a) == 'float':
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
    # vetor dos ruidos brancos
    e = distribuicao.rvs(t)
    # criando o y
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
    # vetor dos ruidos brancos
    e = distribuicao.rvs(t)
    # criando o y
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
    # vetor dos ruidos brancos
    e = distribuicao.rvs(t)
    # criando o y
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


def estima_ar_momentos(serie, p):
  T = len(serie)
  auto_corr =  np.zeros(T)
  rho_hat = np.zeros(p)
  B_hat = np.zeros((p, p))
  for i in range(p):
    auto_corr[i] =  autocorr(serie, i)

  for i in range(p):
    rho_hat[i] = auto_corr[i+1]
    parte1 = vetor[i::-1]
    k1 = len(parte1)
    parte2 = vetor[:(p-k1)]
    B_hat[i,:] = np.append(parte1, parte2)

  phi_hat = np.dot(np.linalg.inv(B_hat), rho_hat) 
  return phi_hat

  

def estima_ar_sres(serie, p):
    T = len(serie)
    y = np.zeros(T-p)
    X = np.zeros((T-p, p))
    for i in range(p+1, T+1):
      vetor = []
      print(i)
      y[i-(p+1)] = serie[i-1]
      print(y)
      for j in range((i-2),(i-p-2),-1):
        vetor.append(serie[j])
      print(vetor)
      X[i-(p+1)] = vetor
      print(X)
    
    modelo = regressao1.regressao_linear_multipla(X, y)
    return None



#ar = estima_ar(serie, 2)


# serie = (1.91977621, 1.28220621, -10.28502884, 0.05528268, -7.70927579, -4.07136374, 2.49497288,
#9.95347029, -9.12282773, 8.55841024, 1.83772973, 4.92246366, 0.99161611, 9.48713457,
#0.08879661, -4.10109202, 6.02946628, 3.83192066, 1.81805589, 4.80681391, 1.75589291,
#4.58592555, 3.99840399, -2.16249826, 12.45684188, -4.76299329, 6.66007040, 8.94450078,
#8.86279070, 3.54524527, 10.66194333, 2.13487495, 5.58371194, 4.80771771, 5.69578136,
#0.61408318, 1.19837986, 7.47275806, -3.23850695, -1.64591651, -7.84608081, -9.38808888,
#-10.86801660, -10.71336816, -14.01940271, -20.07366069, -11.57915130, -8.40834226,
#-2,.94101181, -2.54914225, -3.92513403, -9.05114364, 1.25228972, -9.21545138, -2.37872308,
#-6.72475105, -3.91981823, -6.64977937, -9.41822537, 4.30784905, -8.49249926, -3.75532640,
#2.89825597, -11.51260616, -2.08368166, 0.84888206, 6.89654450, 1.79331617, -2.31935388,
#1.21120305, -1.96684798, -0.35056477, -6.33523678, -4.82637861, 3.61987739, 3.80126051,
#0.13477744, 2.31263290, -9.46570795, -4.36913911, -8.42272317, -12.80922117, -10.95459205,
#1.73397932, -2.85671614, 3.63500048, -7.44174490, 9.51805378, -3.51007600, 12.13185966,
#-5.79112968, 11.85855497, -5.34877159, 12.38388102, 1.84714044, 10.07881768, 12.18798107,
# 9.68835807, 8.82703669, 16.04423162)
