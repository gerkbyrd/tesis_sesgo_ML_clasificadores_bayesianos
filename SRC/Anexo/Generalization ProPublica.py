# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:26:44 2020

@author: Mauricio
"""

import numpy as np
from helper import *
import os


np.random.seed(1234)

"""
PRECISIÓN Y ERRORES DE GENERALIZACION, ENTRENAMIENTO Y VALIDACION
"""
X_train = np.load(r"SRC/Anexo/Arreglos ProPublica/X_train.npy")
X_val = np.load(r"SRC/Anexo/Arreglos ProPublica/X_val.npy")
X_test = np.load(r"SRC/Anexo/Arreglos ProPublica/X_test.npy")

X_train_v = np.load(r"SRC/Anexo/Arreglos ProPublica/X_train_v.npy")
X_val_v = np.load(r"SRC/Anexo/Arreglos ProPublica/X_val_v.npy")
X_test_v = np.load(r"SRC/Anexo/Arreglos ProPublica/X_test_v.npy")



y_train = np.load(r"SRC/Anexo/Arreglos ProPublica/y_train.npy")
y_train_v = np.load(r"SRC/Anexo/Arreglos ProPublica/y_train_v.npy")
y_val = np.load(r"SRC/Anexo/Arreglos ProPublica/y_val.npy")
y_val_v = np.load(r"SRC/Anexo/Arreglos ProPublica/y_val_v.npy")
y_test = np.load(r"SRC/Anexo/Arreglos ProPublica/y_test.npy")
y_test_v = np.load(r"SRC/Anexo/Arreglos ProPublica/y_test_v.npy")

coefs_freq = np.load(r"SRC/Anexo/Arreglos ProPublica/coefs_freq.npy")
coefs_freq_v = np.load(r"SRC/Anexo/Arreglos ProPublica/coefs_freq_v.npy")

coefs_opt = np.load(r"SRC/Anexo/Arreglos ProPublica/perf_nbww.npy")
coefs_opt_v = np.load(r"SRC/Anexo/Arreglos ProPublica/perf_nbww_v.npy")

coefs_bay = np.load(r"SRC/Anexo/Arreglos ProPublica/perf_mean_bay.npy")
coefs_bay_v = np.load(r"SRC/Anexo/Arreglos ProPublica/perf_mean_bay_v.npy")

training_errors, validation_errors, test_errors = np.zeros((3,2,2)), np.zeros((3,2,2)), np.zeros((3,2,2))

training_inputs = np.array([X_train, X_train_v])
validation_inputs = np.array([X_val, X_val_v])
test_inputs = np.array([X_test, X_test_v])

training_outputs = np.array([y_train, y_train_v])
validation_outputs = np.array([y_val, y_val_v])
test_outputs = np.array([y_test, y_test_v])

coefs_freq = np.array([coefs_freq, coefs_freq_v])
coefs_opt = np.array([coefs_opt, coefs_opt_v])
coefs_bay = np.array([coefs_bay, coefs_bay_v])

coefs_all = np.array([coefs_freq, coefs_opt, coefs_bay])


training_outputs = np.array([y_train, y_train_v])
validation_outputs = np.array([y_val, y_val_v])
test_outputs = np.array([y_test, y_test_v])


#Error de entrenamiento
for i in range(3):
    for j in range(2):
        X, y, w = training_inputs[j], training_outputs[j], coefs_all[i,j]
        training_errors[i,j] = logreg_cost_accuracy(X, y, w)[0]/len(y), logreg_cost_accuracy(X, y, w)[1]

#Error de validación
for i in range(3):
    for j in range(2):
        X, y, w = validation_inputs[j], validation_outputs[j], coefs_all[i,j]
        validation_errors[i,j] = logreg_cost_accuracy(X, y, w)[0]/len(y), logreg_cost_accuracy(X, y, w)[1]

#Error de prueba
for i in range(3):
    for j in range(2):
        X, y, w = test_inputs[j], test_outputs[j], coefs_all[i,j]
        test_errors[i,j] = logreg_cost_accuracy(X, y, w)[0]/len(y), logreg_cost_accuracy(X, y, w)[1]

"""
PRINTING:
    
ERRORS
for x in range(training_errors.shape[0]):
    print(str(x)+":")
    for y in range(training_errors.shape[1]):
        #change this line to val or test when getting those
        print(round(training_errors[x,y,0], 4))
    print("\n")
    
ACCURACIES
for x in range(training_errors.shape[0]):
    print(str(x)+":")
    for y in range(training_errors.shape[1]):
        print(round(training_errors[x,y,1], 3))
    print("\n")
"""






