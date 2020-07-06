# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:13:52 2019

@author: Mauricio
"""

#import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from helper import *
from timeit import default_timer as timer
import os
#tf.enable_eager_execution()


tfd = tfp.distributions

learning_rate = 0.01
max_steps = 1500
batch_size = 32

num_monte_carlo = 50000

tf.reset_default_graph()
tf.compat.v1.random.set_random_seed(1234)
np.random.seed(1234)

logreg_input_matrix = np.load(r"SRC/ProPublica/Arreglos/lgrg_in_bias_v.npy")
logreg_targets = np.reshape(np.load(r"SRC/ProPublica/Arreglos/lgrg_tgt_v.npy"), (4020,1))
coefs_freq = np.load(r"SRC/ProPublica/Arreglos/coefs_freq_v.npy")

#para el modelo nulo (AIC, chi cuadrda, etc):
#logreg_input_matrix = np.ones((logreg_input_matrix.shape[0],1))



num_examples = len(logreg_targets)

"""
CÓDIGO PARA RENDIMIENTO Y GENERALIZACIÓN

data = np.concatenate((logreg_input_matrix, logreg_targets), axis = 1)
np.random.shuffle(data)
logreg_input_matrix, logreg_targets = data[:,:12], data[:,12].reshape((len(data), 1))

X_train = logreg_input_matrix[0:int(num_examples*.8) + 1,:]
X_val = logreg_input_matrix[int(num_examples*.8) + 1:int(num_examples*.9)+1,:]
X_test = logreg_input_matrix[int(num_examples*.9) + 1:num_examples + 1,:]
#X_train1 = tf.convert_to_tensor(X_train)
#X_val = tf.convert_to_tensor(logreg_input_matrix[int(num_examples*.8) + 1:int(num_examples*.9)+1,:])
#X_test = tf.convert_to_tensor(logreg_input_matrix[int(num_examples*.9) + 1:num_examples + 1,:])

y_train = logreg_targets[0:int(num_examples*.8) + 1,:]
y_val = logreg_targets[int(num_examples*.8) + 1:int(num_examples*.9)+1,:]
y_test = logreg_targets[int(num_examples*.9) + 1:num_examples + 1,:]

logreg_input_matrix = X_train
logreg_targets = y_train
"""
X = logreg_input_matrix
y = logreg_targets




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
"""
features, labels = build_input_pipeline(X_train, y_train, batch_size)
"""
features, labels = build_input_pipeline(X, y, batch_size)
# Define a logistic regression model as a Bernoulli distribution
# parameterized by logits from a single linear layer. We use the Flipout
# Monte Carlo estimator for the layer: this enables lower variance
# stochastic gradients than naive reparameterization.

with tf.compat.v1.name_scope("logistic_regression", values=[features]):
  layer = tfp.layers.DenseFlipout(
      units=1,
      activation= None,
      kernel_prior_fn = custom_multivariate_normal_fn(loc = None, scale = np.sqrt(1)),
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
predictions = tf.cast(logits > 0, dtype=tf.int32)
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
      
      if step % 100 == 0:
       loss_value, accuracy_value = sess.run([elbo_loss, accuracy])
       print("Step: {:.3f} Loss: {:.3f} Accuracy: {:.3f}".format(
               step, loss_value, accuracy_value))
       
    end = timer()
    time_bay = round((end-start)*1000, 3)
    start = timer()
    for step in range(max_steps):
      _ = sess.run([train_op1, accuracy_update_op1])
      
      if step % 100 == 0:
       loss_value1, accuracy_value1 = sess.run([elbo_loss1, accuracy1])
       print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
               step, loss_value1, accuracy_value1))
       
    end = timer()
    time_opt = round((end-start)*1000, 3)
           
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
    
    var, pvar, mn, pmn, nbww = sess.run((variance, prior_variance, mean, prior_mean, non_bay_ww))


  
    sqetrf, sqetro, sqetrb, sqevf, sqevo, sqevb, sqetf, sqeto, sqetb = sess.run((sqerr_train_freq, sqerr_train_opt, sqerr_train_bay, 
                                                                                 sqerr_val_freq, sqerr_val_opt, sqerr_val_bay, 
                                                                                 sqerr_tst_freq, sqerr_tst_opt, sqerr_tst_bay)) 
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

ORB_black_samples = GetOddsRatio(ws[:, 0], ws[:, 4])
ORB_woman_samples = GetOddsRatio(ws[:, 0], ws[:, 1])
ORB_u25_samples = GetOddsRatio(ws[:, 0], ws[:, 3])

ORBp_black_samples = GetOddsRatio(ps[:, 0], ps[:, 4])
ORBp_woman_samples = GetOddsRatio(ps[:, 0], ps[:, 1])
ORBp_u25_samples = GetOddsRatio(ps[:, 0], ps[:, 3])

ORo_black = GetOddsRatio(nbww[0], nbww[4])
ORo_woman = GetOddsRatio(nbww[0], nbww[1])
ORo_u25 = GetOddsRatio(nbww[0], nbww[3])


OR_black = GetOddsRatio(coefs_freq[0], coefs_freq[4])
OR_woman = GetOddsRatio(coefs_freq[0], coefs_freq[1])
OR_u25 = GetOddsRatio(coefs_freq[0], coefs_freq[3])

"""
Printing out:
    
Real Sample MIN/MAX
for x in ORB_woman_samples:
    print(str(round(np.mean(x), 6)))
    print(str(round(np.var(x), 6)))
    print(str(round(np.min(x), 6)))
    print(str(round(np.max(x), 6)))
   
3 std dev MIN/MAX
for x in ORB_u25_samples:
    print(str(round(np.mean(x), 6)))
    print(str(round(np.var(x), 6)))
    print(str(round(np.mean(x) - 3*np.sqrt(np.var(x)), 6)))
    print(str(round(np.mean(x) + 3*np.sqrt(np.var(x)), 6)))
    
PERCENTILE MIN/MAX

for x in ORB_u25_samples:
      print(str(round(np.percentile(x, 0.15), 6)))
      print(str(round(np.percentile(x, 99.85), 6)))
      
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

means1 = means[[12,3,5,7]]
variances1 = variances[[12,3,5,7]]
#range ~ -3:4
points = np.linspace(means - 5*np.sqrt(variances), means + 5*np.sqrt(variances), 100)

points1 = np.linspace(means1 - 5*np.sqrt(variances1), means1 + 5*np.sqrt(variances1), 100)

#neg = np.linspace(np.ones(len(means))*-3, means - 3*np.sqrt(variances), 10)
#pos = np.linspace(means + 3*np.sqrt(variances), np.ones(len(means))*4, 10)
#points = np.concatenate((neg, points, pos), axis = 0)
pdfs = np.zeros((len(means), len(points)))
with tf.compat.v1.Session() as sess:
    for x in range(len(means)):
        d = tfd.Normal(loc = means[x], scale = np.sqrt(variances[x]))
        pdfs[x] = sess.run(d.prob(points[:, x]))
        
pdfs1 = np.zeros((len(means1), len(points1)))
with tf.compat.v1.Session() as sess:
    for x in range(len(means1)):
        d = tfd.Normal(loc = means1[x], scale = np.sqrt(variances1[x]))
        pdfs1[x] = sess.run(d.prob(points1[:, x]))
        
labels = np.array(['Constante $b$',
'Factor de género: femenino',
'Factor de edad: mayor de 45',
'Factor de edad: menor de 25',
'Factor de raza: afroamericano',
'Factor de raza: asiático',
'Factor de raza: hispano',
'Factor de raza: nativo americano',
'Factor de raza: otro',
'Conteo de antecedentes',
'Factor de crimen: delito menor',
'Reincidencia en dos años',
'distribución a priori'])
    
labels1 = np.array(['distribución a priori',
'Factor de edad: menor de 25',
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
colors[5], colors[6], colors[7] = col.to_rgba_array('grey'), col.to_rgba_array('black'), col.to_rgba_array('pink')



ax = fig.add_subplot(1, 1, 1)
ax.set_prop_cycle('color', colors)
#ax.set_prop_cycle(color = ['black', 'orange', 'red', 'blue'])
 
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
    
 

 
ax.set_xlim([-3, 4])
ax.set_ylim([-0.1, 10.])#np.max(pdfs)])
#ax.set_ylim([-0.1, 5.2])#np.max(pdfs1)])
ax.vlines(means, -0.1, 12., linestyles='dashed', lw = 1.5)#np.max(pdfs)])
#ax.vlines(means1, -0.1, 6., linestyles='dashed', lw = 1.5)#np.max(pdfs)])
#ax.hlines(0., -3., 4., linestyles='solid', lw = 1.)

ax.set_title('Coeficientes, sus distribuciones a posteriori, y la distribución común a priori (reincidencia violenta)')
#ax.set_title('Coeficientes de alta incertidumbre (reincidencia violenta)')
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
fname=os.path.join(r"SRC/ProPublica/Figuras/",
                                         "PDF_Violent_mod.png")

fname=os.path.join(r"SRC/ProPublica/Figuras/",
                                         "PDF_Violent_inc_mod.png")


canvas.print_figure(fname, format="png")
print("saved {}".format(fname))
