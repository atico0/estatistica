import numpy as np
import pandas as pd
import scipy.stats as sts

#AED

def contar_unicos(x):
  """
  retorna um dicionário onde as chaves são os valores unicos de x
  e os valores são as freqências de cada valor de x

  x: vetor numérico
  vetor do qual serão calculadas as frequências dos valores
  """
  
  dicio = {}
  for i in range(len(x)):
    try:
      dicio[x[i]] += 1
    except:
      dicio[x[i]] += 0
  return dicio



#não paramétrica

def ordena(vetor, modo='cre'):
  """
  ordena os vetores de modo crescente ou decrescente

  vetor: vetor numéricpo
  vetor que será ordenado

  modo: string
  determina se vai ser crescente(cre) ou decrescente(decre)
  """
  for i in range(len(vetor)):
    if i > 0:
      if modo == 'decre':
        if vetor[i] > vetor[i-1]:
          cont = i
          while (vetor[cont] > vetor[cont - 1]) and cont > 0:
            menor = vetor[cont - 1]
            maior = vetor[cont]
            vetor[cont - 1], vetor[cont] = maior, menor
            cont -= 1
      
      if modo == 'cre':
        if vetor[i] < vetor[i-1]:
          cont = i
          while (vetor[cont] < vetor[cont - 1]) and cont > 0:
            menor = vetor[cont]
            maior = vetor[cont - 1]
            vetor[cont], vetor[cont - 1] = maior, menor
            cont -= 1
  return vetor


def ordena_com_origem(vetor, modo='cre'):
  """
  ordena os vetores de modo crescente ou decrescente
  mas kda valor tem além do valor original, também tem o indice original

  vetor: vetor numéricpo
  vetor que será ordenado

  modo: string
  determina se vai ser crescente(cre) ou decrescente(decre)
  """
  novo_vetor = []
  for i in range(len(vetor)):
    novo_vetor.append([vetor[i],i])

  for i in range(len(vetor)):
    if i > 0:


      if modo == 'decre':
        if novo_vetor[i][0] > novo_vetor[i-1][0]:
          cont = i
          while (novo_vetor[cont][0] > novo_vetor[cont - 1][0]) and cont > 0:
            menor = novo_vetor[cont - 1]
            maior = novo_vetor[cont]
            novo_vetor[cont - 1], novo_vetor[cont] = maior, menor
            cont -= 1


      if modo == 'cre':
        if novo_vetor[i][0] < novo_vetor[i-1][0]:
          cont = i
          while (novo_vetor[cont][0] < novo_vetor[cont - 1][0]) and cont > 0:
            maior = novo_vetor[cont - 1]
            menor = novo_vetor[cont]
            novo_vetor[cont - 1], novo_vetor[cont] = menor, maior
            cont -= 1


  return novo_vetor


def ordena_com_origem_e_atual(vetor, modo='cre'):
   """
  ordena os vetores de modo crescente ou decrescente
  mas kda valor tem além do valor original, também tem o indice original e atual

  vetor: vetor numéricpo
  vetor que será ordenado

  modo: string
  determina se vai ser crescente(cre) ou decrescente(decre)
  """
  novo_vetor = ordena_com_origem(vetor=vetor, modo=modo)
  for i in range(len(novo_vetor)):
    novo_vetor[i].append(i)
  return novo_vetor


def ordena_por_elemento(vetor, indice=0, modo='cre'):
  """
  pega uma lista de listas e ordena com base em uma coluna especifica
  
  vetor: vetor numéricpo
  vetor que será ordenado

  indice: valor inteiro
  valor do indice (coluna) com base no qual o vetor será ordenado
  modo: string
  determina se vai ser crescente(cre) ou decrescente(decre)
  """

  for i in range(len(vetor)):
    if i > 0:


      if modo == 'decre':
        if vetor[i][indice] > vetor[i-1][indice]:
          cont = i
          while (vetor[cont][indice] > vetor[cont - 1][indice]) and cont > 0:
            menor = vetor[cont - 1]
            maior = vetor[cont]
            vetor[cont - 1], vetor[cont] = maior, menor
            cont -= 1
      
      if modo == 'cre':
        if vetor[i][indice] < vetor[i-1][indice]:
          cont = i
          while (vetor[cont][indice] < vetor[cont - 1][indice]) and cont > 0:
            menor = vetor[cont]
            maior = vetor[cont - 1]
            vetor[cont], vetor[cont - 1] = maior, menor
            cont -= 1
  return vetor



def arg_grupo(grupos, nomes, modo='cre'):
  """
  pega uma matriz com valores e seus respectivos grupos e ordena com base no valor

  grupos: vetor numérico
  vetor que vai ser usado para ordenar os grupos

  nomes: vetor de strings
  vetor com os nomes dos grupos a que pertencem os valores
  """
  lista = []
  for i in range(len(grupos)):
    grupo = grupos[i]
    nome = nomes[i]
    for valor in grupo:
      lista.append([valor, nome])
  lista = ordena_por_elemento(lista, indice=0, modo=modo)
  return lista


def rank_Siegel_Tukey(vetor):
  """
  função para o rankeamento no teste de Siegel Tukey para variância

  vetor: vetor numérico
  vetor que será ranqueado

  funcionamento: o ranque 1 vai para o menor valor
  dai os ranques vão alternando entre os maiores e os menores (2 por vez)

  empates: em caso de empates a alternância não acontece até que os empates
  acabam e os ranks dos empates são as médias dos ranks
  (a maior parte das linhas de códigos dessa func são pra resolver essa questão)
  """

  ranks = []
  
  maior_menor = 1
  cont_troca = 0
  indice_menor = 0
  indice_maior = len(vetor)
  rank = 1
  caso = ''
  for i in range(len(vetor)):
    soma_empates = 0
    cont_empates = 0

    if i == 0:
      if vetor[indice_menor + 1][0] == vetor[indice_menor][0]:
        inicio = indice_menor
        while vetor[indice_menor + 1][0] == vetor[indice_menor][0]:
          soma_empates += rank
          cont_empates += 1
          rank += 1
          indice_menor += 1
          if vetor[indice_menor + 1][0] != vetor[indice_menor][0]:
            fim = indice_menor
            media_empates = soma_empates/cont_empates
            caso = 'primeiro'
      else:
        ranks.append([1, vetor[i][1]])
        rank += 1
        indice_menor += 1
    
    if caso =='primeiro':
      if (inicio <= i <= fim):
        ranks.append([media_empates, vetor[i][1]])
        maior_menor = -1
      if i == fim:
        caso = ''
    elif caso =='menor':
      if (inicio <= i <= fim):
        ranks.append([media_empates, vetor[i][1]])
        maior_menor = -1
      if i == fim:
        caso = ''
    elif caso == 'maior':
      if (inicio >= i >= fim):
        ranks.append([media_empates, vetor[i][1]])
        maior_menor = -1
      if i == fim:
        caso = ''
        



          

      ranks.append([rank, vetor[indice_menor][1]])
      rank += 1
      indice_menor +=1

    if maior_menor == 1:
      if vetor[indice_menor + 1][0] == vetor[indice_menor][0]:
        inicio = indice_menor
        while vetor[indice_menor + 1][0] == vetor[indice_menor][0]:
          soma_empates += rank
          cont_empates += 1
          rank += 1
          indice_menor += 1
          if vetor[indice_menor + 1][0] != vetor[indice_menor][0]:
            fim = indice_menor
            media_empates = soma_empates/cont_empates
            caso = 'menor'
      else:
        ranks.append([rank, vetor[indice_menor][1]])

        cont_troca +=1
        indice_menor +=1
        rank += 1
        if cont_troca ==2:
          maior_menor = 0

    elif maior_menor == 0:
      if vetor[indice_maior - 1][0] == vetor[indice_maior][0]:
        inicio = indice_maior
        while vetor[indice_maior - 1][0] == vetor[indice_maior][0]:
          soma_empates += rank
          cont_empates += 1
          rank += 1
          indice_maior -= 1
          if vetor[indice_maior - 1][0] != vetor[indice_maior][0]:
            fim = indice_maior
            media_empates = soma_empates/cont_empates
            caso = 'maior'
      else:
        ranks.append([rank, vetor[indice_maior][1]])
        
        cont_troca +=1
        indice_menor -=1
        rank += 1
        if cont_troca == 2:
          maior_menor = 1
    else:
      if caso == 'primeiro':
        maior_menor = 0
      if caso == 'menor':
        maior_menor = 1
      if caso == 'maior':
        maior_menor = 0

  return ranks


def diz_ordem(vetor, modo='cre'):
  """
  funciona como o arg_sort (ou como eu acho que devia funcionar)

  vetor: vetor numérico
  vetor no qual os indices serão calculados
  """
  vetor_auxiliar = ordena_com_origem_e_atual(vetor=vetor, modo=modo)
  novo_vetor = [0]*len(vetor)
  for c in vetor_auxiliar:
    indice = c[1]
    novo_vetor[indice] = c[2]
  return novo_vetor


def sorteio(vetor):  
  """eu tive que criar essa função pq o np.arsort não estava funcionando direito
  (ou como eu acho que devia funcionar)
  é igual a função diz_ordem mas sem a opção de modo mas vou dxa só pra lembrar
  
  vetor: vetor numérico
  vetor no qual os indices serão calculados
  """
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
