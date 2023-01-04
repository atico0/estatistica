# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 18:37:11 2022

@author: luisg
"""

from scipy import stats as sts
import math
import numpy as np
import pandas as pd



class Teste:
    def __init__(self, param=False, alpha=0.05, param_co=False,
                 h0=False, h1=False):
        self.param = param
        self.h0 = h0
        self.h1 = h1
        self.alpha = alpha
        
    def calc_p_valor():
        pass
    def calculando_estatistica(vetor):
        pass



class Independencia(Teste):
    
    
    def calc_est(self, vetor):
        self.dim = vetor.shape
        N = np.sum(vetor)
        ni_ = np.sum(vetor,axis=1)
        n_j = np.sum(vetor,axis=0)
        
        soma_nij_log = 0
        for i in range(vetor.shape[0]):
            for j in range(vetor.shape[1]):
                soma_nij_log += (vetor[i,j] * math.log(vetor[i,j]))
                
        soma_ni_log = 0
        soma_nj_log = 0
        
        for i in range(len(ni_)):
            soma_ni_log += (ni_[i]*math.log(ni_[i]))
            
        for j in range(len(n_j)):
            soma_nj_log += (n_j[j]*math.log(n_j[j]))
        N_log = N * math.log(N)
        self.est = -2 * (soma_ni_log + soma_nj_log - N_log - soma_nij_log)
        return self.est
    
    
    def calc_c(self):
        self.gl = (self.dim[0] - 1) * (self.dim[1] - 1)
        chi2 = sts.chi2(df=self.gl)
        self.c0 = chi2.ppf(1-self.alpha)
        return self.c0
    
    def decisao(self):
        if self.est > self.c0:
            print(f'Como {self.est} pertence a região de rejeição')
            print(f'>{self.c0}, então nós rejeitamos H0 a um nível de')
            print(f'significância de {self.alpha*100}%')
        else:
            print(f'Como {self.est} não pertence a região de rejeição')
            print(f'>{self.c0}, então nós não rejeitamos H0 a um nível de')
            print(f'significância de {self.alpha*100}%')
    def calc_p_valor(self):
        chi2 = sts.chi2(df=self.gl)
        self.p_value = 1 - chi2.cdf(self.est)
        return(self.p_value)



vetor = np.array([[32, 26, 19],
                  [33, 60, 66],
                  [21, 29, 120]])

teste = Independencia()
teste.calc_est(vetor)
teste.est
teste.calc_c()
teste.c0
teste.decisao()
teste.calc_p_valor()

vetor = np.array([[50, 26],
                  [48, 76]])
teste.calc_est(vetor)
teste.est


dir(sts)
dir(sts.norm)

dir(math)
?math.log

vetor = np.array([[1,2],
                  [3,4]])
np.sum(vetor)
np.sum(vetor,axis=0)
np.sum(vetor,axis=1)

?sts.chi2


sts.chi2(df=1).cdf(1)











