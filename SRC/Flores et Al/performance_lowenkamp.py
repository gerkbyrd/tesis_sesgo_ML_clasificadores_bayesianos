# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:04:31 2020

@author: Mauricio
"""

#import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from helper import *
import os
#tf.enable_eager_execution()

tfd = tfp.distributions
tf.reset_default_graph()
tf.compat.v1.random.set_random_seed(1234)
np.random.seed(1234)
"""
LL, ACC = logreg_cost_accuracy(X, y, w)
CHI SQUARE = -2*(LL0 - LL)
LL = LL
PSEUDO R2 = 1 - (LL/LL0)

AIC:
DEV = 2*LL
AIC = 2*LL + 2*K (K = n parametros)
"""
logreg_input_matrix1 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input1.npy")
logreg_input_matrix2 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input2.npy")
logreg_input_matrix3 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input3.npy")
logreg_input_matrix4 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input4.npy")
logreg_input_matrix0 = np.ones((logreg_input_matrix1.shape[0],1))



logreg_input_matrix1v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input_v1.npy")
logreg_input_matrix2v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input_v2.npy")
logreg_input_matrix3v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input_v3.npy")
logreg_input_matrix4v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input_v4.npy")
logreg_input_matrix0v = np.ones((logreg_input_matrix1v.shape[0],1))


logreg_input_matrix1vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input_vnf1.npy")
logreg_input_matrix2vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input_vnf2.npy")
logreg_input_matrix3vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input_vnf3.npy")
logreg_input_matrix4vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input_vnf4.npy")
logreg_input_matrix0vnf = np.ones((logreg_input_matrix1vnf.shape[0],1))

logreg_targets = np.reshape(np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\lgrg_trgt_Lowenkamp.npy"), (5278,1)).astype('float32')
logreg_targets_v = np.reshape(np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\lgrg_tgt_Lowenkamp_v.npy"), (len(logreg_input_matrix1v),1)).astype('float32')
logreg_targets_vnf = np.reshape(np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\lgrg_tgt_Lowenkamp_vnf.npy"), (len(logreg_input_matrix1vnf),1)).astype('float32')


null = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\null.npy")
null_v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\null_v.npy")
null_vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\null_vnf.npy")

null_bay = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\null_bay.npy")
null_bay_v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\null_v_bay.npy")
null_bay_vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\null_vnf_bay.npy")

null_opt = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\null_opt.npy")
null_opt_v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\null_v_opt.npy")
null_opt_vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\null_vnf_opt.npy")

coefs_freq1 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs1.npy")
coefs_freq2 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs2.npy")
coefs_freq3 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs3.npy")
coefs_freq4 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs4.npy")

coefs_freq1v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs1_v.npy")
coefs_freq2v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs2_v.npy")
coefs_freq3v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs3_v.npy")
coefs_freq4v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs4_v.npy")

coefs_freq1vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs1_vnf.npy")
coefs_freq2vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs2_vnf.npy")
coefs_freq3vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs3_vnf.npy")
coefs_freq4vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs4_vnf.npy")

coefs_opt1 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww1.npy")
coefs_opt2 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww2.npy")
coefs_opt3 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww3.npy")
coefs_opt4 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww4.npy")

coefs_opt1v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww1v.npy")
coefs_opt2v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww2v.npy")
coefs_opt3v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww3v.npy")
coefs_opt4v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww4v.npy")

coefs_opt1vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww1vnf.npy")
coefs_opt2vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww2vnf.npy")
coefs_opt3vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww3vnf.npy")
coefs_opt4vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\nbww4vnf.npy")

coefs_bay1 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay1.npy")
coefs_bay2 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay2.npy")
coefs_bay3 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay3.npy")
coefs_bay4 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay4.npy")


coefs_bay1v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay1v.npy")
coefs_bay2v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay2v.npy")
coefs_bay3v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay3v.npy")
coefs_bay4v = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay4v.npy")

coefs_bay1vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay1vnf.npy")
coefs_bay2vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay2vnf.npy")
coefs_bay3vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay3vnf.npy")
coefs_bay4vnf = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\mean_bay4vnf.npy")


matrices = []
coefs = []
targets = []
nulls = []

matrices.append([logreg_input_matrix1, logreg_input_matrix2, logreg_input_matrix3, logreg_input_matrix4])
matrices.append([logreg_input_matrix1v, logreg_input_matrix2v, logreg_input_matrix3v, logreg_input_matrix4v])
matrices.append([logreg_input_matrix1vnf, logreg_input_matrix2vnf, logreg_input_matrix3vnf, logreg_input_matrix4vnf])

coefs.append([coefs_freq1, coefs_freq2, coefs_freq3, coefs_freq4])
coefs.append([coefs_freq1v, coefs_freq2v, coefs_freq3v, coefs_freq4v])
coefs.append([coefs_freq1vnf, coefs_freq2vnf, coefs_freq3vnf, coefs_freq4vnf])

targets.append([logreg_targets])
targets.append([logreg_targets_v])
targets.append([logreg_targets_vnf])

nulls.append([null, logreg_input_matrix0])
nulls.append([null_v, logreg_input_matrix0v])
nulls.append([null_vnf, logreg_input_matrix0vnf])

all_measures = []

for i, mtx in enumerate(matrices):
    measures = []
    ww, null, yy = coefs[i], nulls[i], targets[i][0]    
    for j in range(len(mtx)):
        xx = mtx[j]
        NLLm, acc = logreg_cost_accuracy(xx, yy, ww[j])
        NLL0, acc0 = logreg_cost_accuracy(null[1], yy, null[0]) 
        chi2, r2 = -2*(NLLm - NLL0), 1 - (NLLm/NLL0) 
        measures.append([chi2, -NLLm, r2])
    all_measures.append(measures)
 
"""
PRINTER:
for j,x in enumerate(all_measures[1]):
    for k in x:
        print(round(k, 2))
"""
coefs.clear()
nulls.clear()
all_measures.clear()

coefs.append([coefs_bay1, coefs_bay2, coefs_bay3, coefs_bay4])
coefs.append([coefs_bay1v, coefs_bay2v, coefs_bay3v, coefs_bay4v])
coefs.append([coefs_bay1vnf, coefs_bay2vnf, coefs_bay3vnf, coefs_bay4vnf])

nulls.append([null_bay, logreg_input_matrix0])
nulls.append([null_bay_v, logreg_input_matrix0v])
nulls.append([null_bay_vnf, logreg_input_matrix0vnf])

for i, mtx in enumerate(matrices):
    measures = []
    ww, null, yy = coefs[i], nulls[i], targets[i][0]    
    for j in range(len(mtx)):
        xx = mtx[j]
        NLLm, acc = logreg_cost_accuracy(xx, yy, ww[j])
        NLL0, acc0 = logreg_cost_accuracy(null[1], yy, null[0]) 
        chi2, r2 = -2*(NLLm - NLL0), 1 - (NLLm/NLL0) 
        measures.append([chi2, -NLLm, r2])
    all_measures.append(measures)
    
coefs.clear()
nulls.clear()
all_measures.clear()

coefs.append([coefs_opt1, coefs_opt2, coefs_opt3, coefs_opt4])
coefs.append([coefs_opt1v, coefs_opt2v, coefs_opt3v, coefs_opt4v])
coefs.append([coefs_opt1vnf, coefs_opt2vnf, coefs_opt3vnf, coefs_opt4vnf])

nulls.append([null_opt, logreg_input_matrix0])
nulls.append([null_opt_v, logreg_input_matrix0v])
nulls.append([null_opt_vnf, logreg_input_matrix0vnf])

for i, mtx in enumerate(matrices):
    measures = []
    ww, null, yy = coefs[i], nulls[i], targets[i][0]    
    for j in range(len(mtx)):
        xx = mtx[j]
        NLLm, acc = logreg_cost_accuracy(xx, yy, ww[j])
        NLL0, acc0 = logreg_cost_accuracy(null[1], yy, null[0]) 
        chi2, r2 = -2*(NLLm - NLL0), 1 - (NLLm/NLL0) 
        measures.append([chi2, -NLLm, r2])
    all_measures.append(measures)
        

