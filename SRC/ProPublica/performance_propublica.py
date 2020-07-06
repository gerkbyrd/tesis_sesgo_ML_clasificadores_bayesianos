# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:34:41 2020

@author: Mauricio
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from helper import *
import os
#tf.enable_eager_execution()
"""
LL, ACC = logreg_cost_accuracy(X, y, w)
CHI SQUARE = -2*(LL0 - LL)
LL = LL
PSEUDO R2 = 1 - (LL/LL0)

AIC:
DEV = 2*NLL
AIC = 2*NLL + 2*K (K = n parametros)
"""
tfd = tfp.distributions
tf.reset_default_graph()
tf.compat.v1.random.set_random_seed(1234)
np.random.seed(1234)

logreg_input_matrix = np.load(r"SRC/ProPublica/Arreglos/lgrg_in_bias.npy")
logreg_input_matrix_v = np.load(r"SRC/ProPublica/Arreglos/lgrg_in_bias_v.npy")
logreg_input_matrix0 = np.ones((logreg_input_matrix.shape[0],1))
logreg_input_matrix_v0 = np.ones((logreg_input_matrix_v.shape[0],1))

logreg_targets = np.reshape(np.load(r"SRC/ProPublica/Arreglos/lgrg_tgt.npy"), (6172,1))
logreg_targets_v = np.reshape(np.load(r"SRC/ProPublica/Arreglos/lgrg_tgt_v.npy"), (4020,1))

null = np.load(r"SRC/ProPublica/Arreglos/null.npy")
null_v = np.load(r"SRC/ProPublica/Arreglos/null_v.npy")

null_bay = np.load(r"SRC/ProPublica/Arreglos/null_bay.npy")
null_bay_v = np.load(r"SRC/ProPublica/Arreglos/null_v_bay.npy")


coefs_freq = np.load(r"SRC/ProPublica/Arreglos/coefs_freq.npy").reshape((12,1))
coefs_freq_v = np.load(r"SRC/ProPublica/Arreglos/coefs_freq_v.npy").reshape((12,1))

coefs_opt = np.load(r"SRC/ProPublica/Arreglos/nbww.npy")
coefs_opt_v = np.load(r"SRC/ProPublica/Arreglos/nbww_v.npy")

coefs_bay = np.load(r"SRC/ProPublica/Arreglos/mean_bay.npy")
coefs_bay_v = np.load(r"SRC/ProPublica/Arreglos/mean_bay_v.npy")


LL0 = logreg_cost_accuracy(logreg_input_matrix0, logreg_targets, null)[0]
LL = logreg_cost_accuracy(logreg_input_matrix, logreg_targets, coefs_freq)[0]

LLV0 = logreg_cost_accuracy(logreg_input_matrix_v0, logreg_targets_v, null_v)[0]
LLV = logreg_cost_accuracy(logreg_input_matrix_v, logreg_targets_v, coefs_freq_v)[0]

LL0B = logreg_cost_accuracy(logreg_input_matrix0, logreg_targets, null_bay)[0]
LLB = logreg_cost_accuracy(logreg_input_matrix, logreg_targets, coefs_bay)[0]

LLV0B = logreg_cost_accuracy(logreg_input_matrix_v0, logreg_targets_v, null_bay_v)[0]
LLVB = logreg_cost_accuracy(logreg_input_matrix_v, logreg_targets_v, coefs_bay_v)[0]

