# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1

@author: Mauricio
"""


import numpy as np
from helper import *
import os


np.random.seed(1234)

"""
PRECISIÓN Y ERRORES DE GENERALIZACION, ENTRENAMIENTO Y VALIDACION
"""
X_train1 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train1.npy")
X_train2 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train2.npy")
X_train3 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train3.npy")
X_train4 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train4.npy")
X_val1 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val1.npy")
X_val2 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val2.npy")
X_val3 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val3.npy")
X_val4 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val4.npy")
X_test1 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test1.npy")
X_test2 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test2.npy")
X_test3 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test3.npy")
X_test4 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test4.npy")

X_train1v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train1v.npy")
X_train2v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train2v.npy")
X_train3v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train3v.npy")
X_train4v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train4v.npy")
X_val1v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val1v.npy")
X_val2v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val2v.npy")
X_val3v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val3v.npy")
X_val4v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val4v.npy")
X_test1v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test1v.npy")
X_test2v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test2v.npy")
X_test3v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test3v.npy")
X_test4v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test4v.npy")

X_train1vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train1vnf.npy")
X_train2vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train2vnf.npy")
X_train3vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train3vnf.npy")
X_train4vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_train4vnf.npy")
X_val1vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val1vnf.npy")
X_val2vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val2vnf.npy")
X_val3vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val3vnf.npy")
X_val4vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_val4vnf.npy")
X_test1vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test1vnf.npy")
X_test2vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test2vnf.npy")
X_test3vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test3vnf.npy")
X_test4vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/X_test4vnf.npy")


y_train = np.load(r"SRC/Anexo/Arreglos Lowenkamp/y_train1.npy")
y_train_v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/y_train1v.npy")
y_train_vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/y_train1vnf.npy")
y_val = np.load(r"SRC/Anexo/Arreglos Lowenkamp/y_val1.npy")
y_val_v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/y_val1v.npy")
y_val_vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/y_val1vnf.npy")
y_test = np.load(r"SRC/Anexo/Arreglos Lowenkamp/y_test1.npy")
y_test_v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/y_test1v.npy")
y_test_vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/y_test1vnf.npy")

coefs_freq1 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq1.npy")
coefs_freq2 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq2.npy")
coefs_freq3 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq3.npy")
coefs_freq4 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq4.npy")
coefs_freq1v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq1v.npy")
coefs_freq2v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq2v.npy")
coefs_freq3v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq3v.npy")
coefs_freq4v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq4v.npy")
coefs_freq1vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq1vnf.npy")
coefs_freq2vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq2vnf.npy")
coefs_freq3vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq3vnf.npy")
coefs_freq4vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/coefs_freq4vnf.npy")

coefs_opt1 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww1.npy")
coefs_opt2 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww2.npy")
coefs_opt3 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww3.npy")
coefs_opt4 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww4.npy")
coefs_opt1v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww1v.npy")
coefs_opt2v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww2v.npy")
coefs_opt3v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww3v.npy")
coefs_opt4v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww4v.npy")
coefs_opt1vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww1vnf.npy")
coefs_opt2vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww2vnf.npy")
coefs_opt3vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww3vnf.npy")
coefs_opt4vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/nbww4vnf.npy")

coefs_bay1 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay1.npy")
coefs_bay2 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay2.npy")
coefs_bay3 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay3.npy")
coefs_bay4 = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay4.npy")
coefs_bay1v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay1v.npy")
coefs_bay2v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay2v.npy")
coefs_bay3v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay3v.npy")
coefs_bay4v = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay4v.npy")
coefs_bay1vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay1vnf.npy")
coefs_bay2vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay2vnf.npy")
coefs_bay3vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay3vnf.npy")
coefs_bay4vnf = np.load(r"SRC/Anexo/Arreglos Lowenkamp/mean_bay4vnf.npy")


training_errors, validation_errors, test_errors = np.zeros((3,12,2)), np.zeros((3,12,2)), np.zeros((3,12,2))

training_inputs = np.array([X_train1, X_train2, X_train3, X_train4, X_train1v, X_train2v, X_train3v, X_train4v,
                            X_train1vnf, X_train2vnf, X_train3vnf, X_train4vnf])
validation_inputs = np.array([X_val1, X_val2, X_val3, X_val4, X_val1v, X_val2v, X_val3v, X_val4v,
                            X_val1vnf, X_val2vnf, X_val3vnf, X_val4vnf])
test_inputs = np.array([X_test1, X_test2, X_test3, X_test4, X_test1v, X_test2v, X_test3v, X_test4v,
                            X_test1vnf, X_test2vnf, X_test3vnf, X_test4vnf])

training_outputs = np.array([y_train, y_train_v, y_train_vnf])
validation_outputs = np.array([y_val, y_val_v, y_val_vnf])
test_outputs = np.array([y_test, y_test_v, y_test_vnf])

coefs_freq = np.array([coefs_freq1, coefs_freq2, coefs_freq3, coefs_freq4, coefs_freq1v, coefs_freq2v, coefs_freq3v, coefs_freq4v, 
                       coefs_freq1vnf, coefs_freq2vnf, coefs_freq3vnf, coefs_freq4vnf])
coefs_opt = np.array([coefs_opt1, coefs_opt2, coefs_opt3, coefs_opt4, coefs_opt1v, coefs_opt2v, coefs_opt3v, coefs_opt4v, 
                       coefs_opt1vnf, coefs_opt2vnf, coefs_opt3vnf, coefs_opt4vnf])
coefs_bay = np.array([coefs_bay1, coefs_bay2, coefs_bay3, coefs_bay4, coefs_bay1v, coefs_bay2v, coefs_bay3v, coefs_bay4v, 
                       coefs_bay1vnf, coefs_bay2vnf, coefs_bay3vnf, coefs_bay4vnf])

coefs_all = np.array([coefs_freq, coefs_opt, coefs_bay])


training_outputs = np.array([y_train, y_train_v, y_train_vnf])
validation_outputs = np.array([y_val, y_val_v, y_val_vnf])
test_outputs = np.array([y_test, y_test_v, y_test_vnf])

"""
Error de entrenamiento
"""

for i in range(3):
    for j in range(12):
        X, y, w = training_inputs[j], training_outputs[int(j/4)], coefs_all[i,j]
        training_errors[i,j] = logreg_cost_accuracy(X, y, w)[0]/len(y), logreg_cost_accuracy(X, y, w)[1]

for i in range(3):
    for j in range(12):
        X, y, w = validation_inputs[j], validation_outputs[int(j/4)], coefs_all[i,j]
        validation_errors[i,j] = logreg_cost_accuracy(X, y, w)[0]/len(y), logreg_cost_accuracy(X, y, w)[1]

for i in range(3):
    for j in range(12):
        X, y, w = test_inputs[j], test_outputs[int(j/4)], coefs_all[i,j]
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
        #change this line to val or test when getting those
        print(round(test_errors[x,y,1], 3))
    print("\n")
"""