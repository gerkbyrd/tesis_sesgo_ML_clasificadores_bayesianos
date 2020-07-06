# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:35:38 2020

@author: Mauricio
"""

#import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from helper import *
import os
from timeit import default_timer as timer
#tf.enable_eager_execution()

tfd = tfp.distributions

learning_rate = 0.01
max_steps = 1500
#para tener resultados m+as estables en el modelo nulo, cambiar iteraciones a 1470...
#max_steps = 1470
batch_size = 32
num_monte_carlo = 50000

tf.reset_default_graph()
tf.compat.v1.random.set_random_seed(1234)
np.random.seed(1234)
 
logreg_input_matrix1 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input1.npy")
logreg_input_matrix2 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input2.npy")
logreg_input_matrix3 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input3.npy")
logreg_input_matrix4 = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\input4.npy")

logreg_targets = np.reshape(np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\lgrg_trgt_Lowenkamp.npy"), (5278,1)).astype('float32')
#coefs_freq = np.reshape(np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\coefs_freq_wb.npy"), (8,1))
#coefs_freq = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\coefs1.npy")
num_examples = len(logreg_targets)

logreg_input_matrix = logreg_input_matrix4

#para modelos nulos (AIC, chi cuadrada, etc)
#logreg_input_matrix0 = np.ones((logreg_input_matrix1.shape[0],1))
#logreg_input_matrix = logreg_input_matrix0

"""
CÓDIGO PARA RENDIMIENTO Y GENERALIZACIÓN

data = np.concatenate((logreg_input_matrix, logreg_targets), axis = 1)
np.random.shuffle(data)
logreg_input_matrix, logreg_targets = data[:,:(data.shape[1] - 1)], data[:,(data.shape[1] - 1)].reshape((len(data), 1))

X_train = logreg_input_matrix[0:int(num_examples*.8) + 1,:]
X_val = logreg_input_matrix[int(num_examples*.8) + 1:int(num_examples*.9)+1,:]
X_test = logreg_input_matrix[int(num_examples*.9) + 1:num_examples + 1,:]

y_train = logreg_targets[0:int(num_examples*.8) + 1,:]
y_val = logreg_targets[int(num_examples*.8) + 1:int(num_examples*.9)+1,:]
y_test = logreg_targets[int(num_examples*.9) + 1:num_examples + 1,:]

logreg_input_matrix = X_train
logreg_targets = y_train
"""
X, y = logreg_input_matrix, logreg_targets



def build_input_pipeline(x, y, batch_size):
  """Build a Dataset iterator for supervised classification.
  Args:
    x: Numpy `array` of features, indexed by the first dimension.
    y: Numpy `array` of labels, with the same first dimension as `x`.
    batch_size: Number of elements in each training batch.
  Returns:
    batch_features: `Tensor` feed  features, of shape
      `[batch_size] + x.shape[1:]`.
    batch_labels: `Tensor` feed of labels, of shape
      `[batch_size] + y.shape[1:]`.
  """
  training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
  training_batches = training_dataset.repeat().batch(batch_size)
  training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)
  batch_features, batch_labels = training_iterator.get_next()
  return batch_features, batch_labels
  
# Generate (and visualize) a toy classification dataset.
#w_true, b_true, x, y = toy_logistic_data(num_examples, 2)
#features, labels = build_input_pipeline(x, y, batch_size)
features, labels = build_input_pipeline(X, y, batch_size)

# Define a logistic regression model as a Bernoulli distribution
# parameterized by logits from a single linear layer. We use the Flipout
# Monte Carlo estimator for the layer: this enables lower variance
# stochastic gradients than naive reparameterization.

with tf.compat.v1.name_scope("logistic_regression", values=[features]):
  layer = tfp.layers.DenseFlipout(
      units=1,
      activation= None,
      kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(),
      bias_posterior_fn = None)
  logits = layer(features)
  labels_distribution = tfd.Bernoulli(logits=logits)
  


with tf.compat.v1.name_scope("non_bayesian_logistic_regression", values=[features]):
  layer1 = tf.layers.Dense(
          units=1, activation= None,  use_bias = False)
  logits1 = layer1(features)
  labels_distribution1 = tfd.Bernoulli(logits=logits1)


# Compute the -ELBO as the loss, averaged over the batch size.
neg_log_likelihood = -tf.reduce_mean(
    input_tensor=labels_distribution.log_prob(labels))
kl = sum(layer.losses) / num_examples
elbo_loss = neg_log_likelihood + kl

neg_log_likelihood1 = -tf.reduce_mean(
    input_tensor=labels_distribution1.log_prob(labels))
kl1 = sum(layer1.losses) / num_examples
elbo_loss1 = neg_log_likelihood1 + kl1

# Build metrics for evaluation. Predictions are formed from a single forward
# pass of the probabilistic layers. They are cheap but noisy predictions.
predictions = tf.cast(logits > 0, dtype=tf.int32)#predictions1 = tf.cast(logits1 > 0, dtype=tf.int32)
predictions1 = tf.cast(logits1 > 0, dtype=tf.int32)

accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(
    labels=labels, predictions=predictions)
accuracy1, accuracy_update_op1 = tf.compat.v1.metrics.accuracy(
    labels=labels, predictions=predictions1)

with tf.compat.v1.name_scope("train"):
  optimizer = tf.compat.v1.train.AdamOptimizer(
          learning_rate = learning_rate)
  train_op = optimizer.minimize(elbo_loss)
  train_op1 = optimizer.minimize(elbo_loss1)
  
init_op = tf.group(tf.compat.v1.global_variables_initializer(), 
                   tf.compat.v1.local_variables_initializer())

with tf.compat.v1.Session() as sess:
    sess.run(init_op)

    # Fit the model to data.
    time_bay, time_opt = 0, 0
    start = timer()
    for step in range(max_steps):
      _ = sess.run([train_op, accuracy_update_op])
      """
      if step % 100 == 0:
       loss_value, accuracy_value = sess.run([elbo_loss, accuracy])
       print("Step: {:.3f} Loss: {:.3f} Accuracy: {:.3f}".format(
               step, loss_value, accuracy_value))
       """
    end = timer()
    time_bay = round((end-start)*1000, 3)
    start = timer()
    for step in range(max_steps):
      _ = sess.run([train_op1, accuracy_update_op1])
      """
      if step % 100 == 0:
       loss_value1, accuracy_value1 = sess.run([elbo_loss1, accuracy1])
       print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
               step, loss_value1, accuracy_value1))
       """
    end = timer()
    time_opt = round((end-start)*1000, 3)
    time = [time_opt, time_bay]
    variance = layer.kernel_posterior._variance()
    prior_variance = layer.kernel_prior._variance()
    mean = layer.kernel_posterior._mean()
    prior_mean = layer.kernel_prior._mean()
    non_bay_ww = layer1.kernel
    coefs_bay = mean
    coefs_opt = non_bay_ww
    w_samples = []
    p_samples = []
    w_draw = layer.kernel_posterior.sample()
    p_draw = layer.kernel_prior.sample()
    
    """-------------------------------
    CÓDIGO PARA EJECUCIONES "EAGER"
    ----------------------------------
coefs_bay = np.load(r"C:\-Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\meanbay_coefsN.npy")
coefs_opt = np.load(r"C:\-Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\Lowenkamp arrays\opt_coefsN.npy")
ytp1 = np.array(log_pred_train_bay)
ytp1l = ytp1 > 0.5
ytp1l = ytp1l.astype('float32')
comp = np.array(ytp1l == y_train)
comp = comp.astype('float32')
unique, counts = np.unique(comp, return_counts = True)
dict(zip(unique, counts))

ytp1 = np.array(log_pred_train_opt)
ytp1l = ytp1 > 0.5
ytp1l = ytp1l.astype('float32')
comp = np.array(ytp1l == y_train)
comp = comp.astype('float32')
unique, counts = np.unique(comp, return_counts = True)
dict(zip(unique, counts))

    """
    """    
    pred_train_bay = tf.matmul(X_train1, coefs_bay)
    pred_train_opt = tf.matmul(X_train1, coefs_opt)
    pred_train_freq = tf.matmul(X_train1, coefs_freq)   
    pred_val_bay = tf.matmul(X_val, coefs_bay)
    pred_val_opt = tf.matmul(X_val, coefs_opt)
    pred_val_freq = tf.matmul(X_val, coefs_freq)
    pred_tst_bay = tf.matmul(X_test, coefs_bay)
    pred_tst_opt = tf.matmul(X_test, coefs_opt)
    pred_tst_freq = tf.matmul(X_test, coefs_freq)
    
    log_pred_train_freq = 1/(1 + tf.exp(-pred_train_freq))
    log_pred_train_bay = 1/(1 + tf.exp(-pred_train_bay))
    log_pred_train_opt = 1/(1 + tf.exp(-pred_train_opt)) 
    log_pred_val_freq = 1/(1 + tf.exp(-pred_val_freq))
    log_pred_val_bay = 1/(1 + tf.exp(-pred_val_bay))
    log_pred_val_opt = 1/(1 + tf.exp(-pred_val_opt)) 
    log_pred_tst_freq = 1/(1 + tf.exp(-pred_tst_freq))
    log_pred_tst_bay = 1/(1 + tf.exp(-pred_tst_bay))
    log_pred_tst_opt = 1/(1 + tf.exp(-pred_tst_opt)) 
    
    
    sqerr_train_bay = tf.reduce_mean(np.subtract(y_train1, log_pred_train_bay)** 2) 
    sqerr_train_opt = tf.reduce_mean(np.subtract(y_train1, log_pred_train_opt)** 2) 
    sqerr_train_freq = tf.reduce_mean(np.subtract(y_train1, log_pred_train_freq)** 2) 
    sqerr_val_bay = tf.reduce_mean(np.subtract(y_val, log_pred_val_bay) ** 2)
    sqerr_val_opt = tf.reduce_mean(np.subtract(y_val, log_pred_val_opt) ** 2)
    sqerr_val_freq = tf.reduce_mean(np.subtract(y_val, log_pred_val_freq) ** 2)
    sqerr_tst_bay = tf.reduce_mean(np.subtract(y_test, log_pred_tst_bay) ** 2)
    sqerr_tst_opt = tf.reduce_mean(np.subtract(y_test, log_pred_tst_opt) ** 2)
    sqerr_tst_freq = tf.reduce_mean(np.subtract(y_test, log_pred_tst_freq) ** 2)
    """    
    var, pvar, mn, pmn, nbww = sess.run((variance, prior_variance, mean, prior_mean, non_bay_ww))
    
    for i in range(num_monte_carlo):
        w = sess.run(w_draw)
        w_samples.append(w)
        
    for i in range(num_monte_carlo):
        p = sess.run(p_draw)
        p_samples.append(p)
        
ws = np.array(w_samples).reshape(num_monte_carlo, len(mn))
ps = np.array(p_samples).reshape(num_monte_carlo, len(mn))

#modelo 1
ORB_black_samples = GetOddsRatio(ws[:, 0], ws[:, 3])
ORB_woman_samples = GetOddsRatio(ws[:, 0], ws[:, 2])
ORB_age_samples = GetOddsRatio(ws[:, 0], ws[:, 1])
ORB_self_samples = GetOddsRatio(ws[:, 0], ws[:, 0])

#modelo 2
ORB_DEC_samples = GetOddsRatio(ws[:, 0], ws[:, 3])
ORB_woman_samples = GetOddsRatio(ws[:, 0], ws[:, 2])
ORB_age_samples = GetOddsRatio(ws[:, 0], ws[:, 1])
ORB_self_samples = GetOddsRatio(ws[:, 0], ws[:, 0])

#modelo 3
ORB_DEC_samples = GetOddsRatio(ws[:, 0], ws[:, 4])
ORB_black_samples = GetOddsRatio(ws[:, 0], ws[:, 3])
ORB_woman_samples = GetOddsRatio(ws[:, 0], ws[:, 2])
ORB_age_samples = GetOddsRatio(ws[:, 0], ws[:, 1])
ORB_self_samples = GetOddsRatio(ws[:, 0], ws[:, 0])

#modelo 4
ORB_BxDEC_samples = GetOddsRatio(ws[:, 0], ws[:, 5])
ORB_DEC_samples = GetOddsRatio(ws[:, 0], ws[:, 4])
ORB_black_samples = GetOddsRatio(ws[:, 0], ws[:, 3])
ORB_woman_samples = GetOddsRatio(ws[:, 0], ws[:, 2])
ORB_age_samples = GetOddsRatio(ws[:, 0], ws[:, 1])
ORB_self_samples = GetOddsRatio(ws[:, 0], ws[:, 0])


"""
ORBp_black_samples = GetOddsRatio(ps[:, 0], ps[:, 3])
ORBp_woman_samples = GetOddsRatio(ps[:, 0], ps[:, 2])
ORBp_age_samples = GetOddsRatio(ps[:, 0], ps[:, 1])
ORBp_self_samples = GetOddsRatio(ps[:, 0], ps[:, 0])

ORo_black = GetOddsRatio(nbww[0], nbww[4])
ORo_woman = GetOddsRatio(nbww[0], nbww[1])
ORo_u25 = GetOddsRatio(nbww[0], nbww[3])


OR_black = GetOddsRatio(coefs_freq[0], coefs_freq[4])
OR_woman = GetOddsRatio(coefs_freq[0], coefs_freq[1])
OR_u25 = GetOddsRatio(coefs_freq[0], coefs_freq[3])

"""
"""
Printing out:
    
Real Sample MIN/MAX
for x in ORB_black_samples:
    print(str(round(np.mean(x), 6)))
    print(str(round(np.var(x), 6)))
    print(str(round(np.min(x), 6)))
    print(str(round(np.max(x), 6)))
   
3 std dev MIN/MAX
for x in ORB_black_samples:
    print(str(round(np.mean(x), 6)))
    print(str(round(np.var(x), 6)))
    print(str(round(np.percentile(x, 0.15), 6)))
    print(str(round(np.percentile(x, 99.85), 6)))
    break
    
PERCENTILE MIN/MAX

for x in ORB_black_samples:
      
      
for x in ORB_black_samples:     
    print(str(round(np.mean(x) - 3*np.sqrt(np.var(x)), 6)))
    print(str(round(np.mean(x) + 3*np.sqrt(np.var(x)), 6)))
      
    
    
    print(str(np.sum(x < (np.mean(x) - 3*np.sqrt(np.var(x)))) + np.sum(x > (np.mean(x) + 3*np.sqrt(np.var(x))))))
"""
"""
PLOTTING STUFF
"""

means = np.array([mn[x][0] for x in range(len(mn))])
means = np.concatenate((means, np.zeros(1)))
variances = np.array([var[x][0] for x in range(len(var))])
variances = np.concatenate((variances, np.ones(1)))

#means1 = means[[12,5,7]]
#variances1 = variances[[12,5,7]]
#range ~ -3:4
points = np.linspace(means - 5*np.sqrt(variances), means + 5*np.sqrt(variances), 100)

#points1 = np.linspace(means1 - 5*np.sqrt(variances1), means1 + 5*np.sqrt(variances1), 100)

#neg = np.linspace(np.ones(len(means))*-3, means - 3*np.sqrt(variances), 10)
#pos = np.linspace(means + 3*np.sqrt(variances), np.ones(len(means))*4, 10)
#points = np.concatenate((neg, points, pos), axis = 0)
pdfs = np.zeros((len(means), len(points)))
with tf.compat.v1.Session() as sess:
    for x in range(len(means)):
        d = tfd.Normal(loc = means[x], scale = np.sqrt(variances[x]))
        pdfs[x] = sess.run(d.prob(points[:, x]))
"""        
pdfs1 = np.zeros((len(means1), len(points1)))
with tf.compat.v1.Session() as sess:
    for x in range(len(means1)):
        d = tfd.Normal(loc = means1[x], scale = np.sqrt(variances1[x]))
        pdfs1[x] = sess.run(d.prob(points1[:, x]))
 """       
labels = np.array(['Constante $b$',
'Edad',
'Género femenino',
'Raza negra',
'distribución a priori'])
    
labels = np.array(['Constante $b$',
'Edad',
'Género femenino',
'Puntaje COMPAS',
'distribución a priori'])
    
labels = np.array(['Constante $b$',
'Edad',
'Género femenino',
'Raza negra',
'Puntaje COMPAS',
'distribución a priori'])

labels = np.array(['Constante $b$',
'Edad',
'Género femenino',
'Raza negra',
'Puntaje COMPAS',
'Puntaje COMPAS X raza negra',
'distribución a priori'])

labels1 = np.array(['distribución a priori',
'Factor de raza: asiático',
'Factor de raza: nativo americano'
])
import matplotlib.colors as col
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.backends import backend_agg


fig = figure.Figure(figsize=(10, 6))
canvas = backend_agg.FigureCanvasAgg(fig)
colors = plt.cm.Spectral(np.linspace(0,1,len(means)))
#Modelo 1
#colors[2] = col.to_rgba_array('grey')
#Modelo 3,4
#colors[3] = col.to_rgba_array('blue')
#colors[5], colors[6], colors[7] = col.to_rgba_array('grey'), col.to_rgba_array('black'), col.to_rgba_array('pink')



ax = fig.add_subplot(1, 1, 1)
ax.set_prop_cycle('color', colors)
#ax.set_prop_cycle(color = ['black', 'red', 'blue'])
#ax.plot(np.linspace(-7, 7, 100), np.ones(100),lw=1, color="black")  
for i in range(len(means)):
    if i == len(means) - 1:
        ax.plot(points[:, i], pdfs[i], lw=2, label = labels[i])
        break
    ax.plot(points[:, i], pdfs[i], lw=2,  label = labels[i])
"""    
for i in range(len(means1)):
    if i == len(means1) - 1:
        ax.plot(points1[:, i], pdfs1[i], lw=2, label = labels1[i])
        break
    ax.plot(points1[:, i], pdfs1[i], lw=2,  label = labels1[i])    
"""
    
 

 
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-0.1, 15.])
ax.vlines(means, -0.1, 15., linestyles='dashed', lw = 1.5)#np.max(pdfs)])
ax.hlines(0., -2., 2., linestyles='solid', lw = 1.)
ax.set_title('Coeficientes, sus distribuciones a posteriori, y la distribución común a priori')
#ax.set_title('Coeficientes de alta incertidumbre')

"""
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
plt.rcParams['mathtext.bf'] = 'STIXGeneral:bold'
#plt.rcParams['mathtext.it.bf'] = 'STIXGeneral:italic:bold'
"""
ax.set_xlabel('Valor numérico de los coeficientes', fontsize = 12)
ax.set_ylabel('Densidad de probabilidad (PDF)', fontsize = 14)

#ax.set_xticks(np.arange(-7,8,1))
#ax.set_yticks(np.arange(-0.1,1.3,0.1))
ax.xaxis.grid(b=True)
ax.yaxis.grid(b=True)

"""
ax.spines['left'].set_position(('data', 0))
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['top'].set_position(('data', 1.0))
#ax.spines['top'].set_color('none')
ax.set_xlabel('x', labelpad=15, fontsize=14)
ax.set_ylabel('p', labelpad = 150, fontsize=14)
"""

ax.legend()
fname=os.path.join(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob",
                                         "PDF_Lowenkamp_1.png")


canvas.print_figure(fname, format="png")
print("saved {}".format(fname))
#sqetrf, sqetro, sqetrb, sqevf, sqevo, sqevb, sqetf, sqeto, sqetb = sess.run((sqerr_train_freq, sqerr_train_opt, sqerr_train_bay, 
  #                                                                              sqerr_val_freq, sqerr_val_opt, sqerr_val_bay, 
   #                                                                              sqerr_tst_freq, sqerr_tst_opt, sqerr_tst_bay))  
    
#LL, NLL1 = sess.run((neg_log_likelihood, neg_log_likelihood1))
"""
PLOTTING RELATIONSHIP RACE VS COMPAS
"""
b_data = logreg_input_matrix[np.where(logreg_input_matrix[:,3] == 1)]
w_data = logreg_input_matrix[np.where(logreg_input_matrix[:,3] == 0)]
ssize = 200
w_samps = ws[np.random.choice(np.arange(ws.shape[0]), size = ssize, replace = False)].T
p_samps = ps[np.random.choice(np.arange(ps.shape[0]), size = ssize, replace = False)].T

def sigmoid(x):
    return 1/(1 + np.exp(-x))


b_preds = sigmoid(np.matmul(b_data, w_samps))
w_preds = sigmoid(np.matmul(w_data, w_samps))

b_mn_preds = sigmoid(np.matmul(b_data, mn))
w_mn_preds = sigmoid(np.matmul(w_data, mn))

b_res, w_res = np.zeros((ssize, 10)), np.zeros((ssize, 10))
b_mn_res, w_mn_res = np.zeros(10), np.zeros(10)

for x in range(10):
    b_res[:, x] = np.mean(b_preds[np.where(b_data[:,4] == x + 1)], axis = 0)
    b_mn_res[x] = np.mean(b_mn_preds[np.where(b_data[:,4] == x + 1)], axis = 0)
    w_res[:, x] = np.mean(w_preds[np.where(w_data[:,4] == x + 1)], axis = 0)
    w_mn_res[x] = np.mean(w_mn_preds[np.where(w_data[:,4] == x + 1)], axis = 0)
    
    

points = np.arange(11)[1:11]
labels = np.array(['Elementos de raza negra', 'Elementos de raza blanca'])
fig = figure.Figure(figsize=(10, 6))
canvas = backend_agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)


for i, y in enumerate(b_res):
    if i == 0:
        ax.plot(points, np.ones(10)*2000, lw=2, label = labels[0], color = 'red')
        ax.plot(points, b_res[i,:], lw=1, alpha = 0.2, color = 'red')
    else:
        ax.plot(points, b_res[i,:], lw=1, alpha = 0.2, color = 'red')
    
for i, y in enumerate(w_res):
    if i == 0:
        ax.plot(points, np.ones(10)*2000, lw=2, label = labels[1], color = 'blue')
        ax.plot(points, w_res[i,:], lw=1, alpha = 0.2, color = 'blue')
    else:
        ax.plot(points, w_res[i,:], lw=1, alpha = 0.2, color = 'blue')
        
ax.plot(points, b_mn_res, lw=2, color = 'black', ls = 'dashed', label = 'Media para raza negra')
ax.plot(points, w_mn_res, lw=2, color = 'grey', ls = 'dashed', label = 'Media para raza blanca')

    
ax.set_xlim([0, 10.1])
ax.set_ylim([0.09, 0.9])
ax.set_xticks(np.arange(11))

ax.set_title('Probabilidad promedio estimada de reincidencia general por raza')
#ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos filtrados, N = 3377)')
#ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos sin filtrar, N = 3967)')
ax.set_xlabel('Puntaje COMPAS', fontsize = 15)
ax.set_ylabel('Probabilidad promedio', fontsize = 12)

ax.xaxis.grid(b=True)
ax.yaxis.grid(b=True)

ax.legend()
fname=os.path.join(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob",
                                         "FloresFig1Bay.png")
canvas.print_figure(fname, format="png")
print("saved {}".format(fname))

fig = figure.Figure(figsize=(10, 6))
canvas = backend_agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)


        
ax.plot(points, b_mn_res, lw=2, color = 'red', label = 'Media para raza negra')
ax.plot(points, w_mn_res, lw=2, color = 'blue', label = 'Media para raza blanca')

    
ax.set_xlim([0, 10.1])
ax.set_ylim([0.09, 0.9])
ax.set_xticks(np.arange(11))

ax.set_title('Probabilidad promedio estimada de reincidencia general por raza')
#ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos filtrados, N = 3377)')
#ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos sin filtrar, N = 3967)')
ax.set_xlabel('Puntaje COMPAS', fontsize = 15)
ax.set_ylabel('Probabilidad promedio', fontsize = 12)

ax.xaxis.grid(b=True)
ax.yaxis.grid(b=True)

ax.legend(loc = 'upper left')
fname=os.path.join(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob",
                                         "FloresFig1BayMN.png")
canvas.print_figure(fname, format="png")
print("saved {}".format(fname))

"""
PRIOR RELATIONSHIPS
"""
pb_preds = sigmoid(np.matmul(b_data, p_samps))
pw_preds = sigmoid(np.matmul(w_data, p_samps))

pb_mn_preds = sigmoid(np.matmul(b_data, pmn))
pw_mn_preds = sigmoid(np.matmul(w_data, pmn))

pb_res, pw_res = np.zeros((ssize, 10)), np.zeros((ssize, 10))
pb_mn_res, pw_mn_res = np.zeros(10), np.zeros(10)

for x in range(10):
    pb_res[:, x] = np.mean(pb_preds[np.where(b_data[:,4] == x + 1)], axis = 0)
    pb_mn_res[x] = np.mean(pb_mn_preds[np.where(b_data[:,4] == x + 1)], axis = 0)
    pw_res[:, x] = np.mean(pw_preds[np.where(w_data[:,4] == x + 1)], axis = 0)
    pw_mn_res[x] = np.mean(pw_mn_preds[np.where(w_data[:,4] == x + 1)], axis = 0)
    
    

points = np.arange(11)[1:11]
labels = np.array(['Elementos de raza negra', 'Elementos de raza blanca'])
fig = figure.Figure(figsize=(10, 6))
canvas = backend_agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)


for i, y in enumerate(pb_res):
    if i == 0:
        ax.plot(points, np.ones(10)*2000, lw=2, label = labels[0], color = 'red')
        ax.plot(points, pb_res[i,:], lw=1, alpha = 0.2, color = 'red')
    else:
        ax.plot(points, pb_res[i,:], lw=1, alpha = 0.2, color = 'red')
    
for i, y in enumerate(pw_res):
    if i == 0:
        ax.plot(points, np.ones(10)*2000, lw=2, label = labels[1], color = 'blue')
        ax.plot(points, pw_res[i,:], lw=1, alpha = 0.2, color = 'blue')
    else:
        ax.plot(points, pw_res[i,:], lw=1, alpha = 0.2, color = 'blue')
        
ax.plot(points, pb_mn_res, lw=2, color = 'black', ls = 'dashed', label = 'Media para raza negra')
ax.plot(points, pw_mn_res, lw=2, color = 'grey', ls = 'dashed', label = 'Media para raza blanca')

    
ax.set_xlim([0, 10.1])
ax.set_ylim([0.0, 1.0])
ax.set_xticks(np.arange(11))

ax.set_title('Probabilidad promedio estimada de reincidencia general por raza')
#ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos filtrados, N = 3377)')
#ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos sin filtrar, N = 3967)')
ax.set_xlabel('Puntaje COMPAS', fontsize = 15)
ax.set_ylabel('Probabilidad promedio', fontsize = 12)

ax.xaxis.grid(b=True)
ax.yaxis.grid(b=True)

ax.legend(loc = 'upper left')
fname=os.path.join(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob",
                                         "FloresFig1Bay_prior.png")
canvas.print_figure(fname, format="png")
print("saved {}".format(fname))

fig = figure.Figure(figsize=(10, 6))
canvas = backend_agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)


        
ax.plot(points, pb_mn_res, lw=2, color = 'red', label = 'Media para raza negra', alpha = 0.7)
ax.plot(points, pw_mn_res, lw=1, color = 'blue', label = 'Media para raza blanca', alpha = 0.7)

    
ax.set_xlim([0, 10.1])
ax.set_ylim([0.09, 0.9])
ax.set_xticks(np.arange(11))

ax.set_title('Probabilidad promedio estimada de reincidencia general por raza')
#ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos filtrados, N = 3377)')
#ax.set_title('Probabilidad promedio estimada de reincidencia violenta por raza (datos sin filtrar, N = 3967)')
ax.set_xlabel('Puntaje COMPAS', fontsize = 15)
ax.set_ylabel('Probabilidad promedio', fontsize = 12)

ax.xaxis.grid(b=True)
ax.yaxis.grid(b=True)

ax.legend(loc = 'upper left')
fname=os.path.join(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob",
                                         "FloresFig1BayMN_prior.png")
canvas.print_figure(fname, format="png")
print("saved {}".format(fname))




 