# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:13:52 2019

@author: Mauricio
"""

#import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import os
#tf.enable_eager_execution()

tfd = tfp.distributions

learning_rate = 0.01
max_steps = 1500
batch_size = 32
num_examples = 6172
num_monte_carlo = 50

logreg_input_matrix = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\lgrg_in_bias.npy")
logreg_targets = np.reshape(np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\lgrg_tgt.npy"), (6172,1))
coefs_freq = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\coefs_freq.npy")
   


X_train = logreg_input_matrix[0:int(num_examples*.8) + 1,:]
X_train1 = tf.convert_to_tensor(X_train)
X_val = tf.convert_to_tensor(logreg_input_matrix[int(num_examples*.8) + 1:int(num_examples*.9)+1,:])
X_test = tf.convert_to_tensor(logreg_input_matrix[int(num_examples*.9) + 1:num_examples + 1,:])

y_train = logreg_targets[0:int(num_examples*.8) + 1,:]
y_train1 = tf.convert_to_tensor(y_train)
y_val = tf.convert_to_tensor(logreg_targets[int(num_examples*.8) + 1:int(num_examples*.9)+1,:])
y_test = tf.convert_to_tensor(logreg_targets[int(num_examples*.9) + 1:num_examples + 1,:])
  
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
features, labels = build_input_pipeline(X_train, y_train, batch_size)

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
    for step in range(max_steps):
      _ = sess.run([train_op, accuracy_update_op])
      if step % 100 == 0:
       loss_value, accuracy_value = sess.run([elbo_loss, accuracy])
       print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
               step, loss_value, accuracy_value))
       
    for step in range(max_steps):
      _ = sess.run([train_op1, accuracy_update_op1])
      if step % 100 == 0:
       loss_value1, accuracy_value1 = sess.run([elbo_loss1, accuracy1])
       print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
               step, loss_value1, accuracy_value1))
       
    variance = layer.kernel_posterior._variance()
    prior_variance = layer.kernel_prior._variance()
    mean = layer.kernel_posterior._mean()
    prior_mean = layer.kernel_prior._mean()
    non_bay_ww = layer1.kernel
    coefs_bay = mean
    coefs_opt = non_bay_ww
    
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
    
    
    sqerr_train_bay = tf.reduce_mean(np.subtract(y_train1, log_pred_train_bay)) ** 2
    sqerr_train_opt = tf.reduce_mean(np.subtract(y_train1, log_pred_train_opt)) ** 2
    sqerr_train_freq = tf.reduce_mean(np.subtract(y_train1, log_pred_train_freq)) ** 2
    sqerr_val_bay = tf.reduce_mean(np.subtract(y_val, log_pred_val_bay)) ** 2
    sqerr_val_opt = tf.reduce_mean(np.subtract(y_val, log_pred_val_opt)) ** 2
    sqerr_val_freq = tf.reduce_mean(np.subtract(y_val, log_pred_val_freq)) ** 2
    sqerr_tst_bay = tf.reduce_mean(np.subtract(y_test, log_pred_tst_bay)) ** 2
    sqerr_tst_opt = tf.reduce_mean(np.subtract(y_test, log_pred_tst_opt)) ** 2
    sqerr_tst_freq = tf.reduce_mean(np.subtract(y_test, log_pred_tst_freq)) ** 2
    
    var, pvar, mn, pmn, nbww = sess.run((variance, prior_variance, mean, prior_mean, non_bay_ww))
    
    
    sqetrf, sqetro, sqetrb, sqevf, sqevo, sqevb, sqetf, sqeto, sqetb = sess.run((sqerr_train_freq, sqerr_train_opt, sqerr_train_bay, 
                                                                                 sqerr_val_freq, sqerr_val_opt, sqerr_val_bay, 
                                                                                 sqerr_tst_freq, sqerr_tst_opt, sqerr_tst_bay))  
   

"""
    # Visualize some draws from the weights prior and posterior.
   
    wp_draw = layer.kernel_prior.sample()
    bp_draw = tf.cast(np.array([0]), dtype = tf.float32)
    candidate_w_bs = []
    for _ in range(num_monte_carlo):
      w, b = sess.run((wp_draw, bp_draw))
      candidate_w_bs.append((w, b))
    visualize_decision(x, y, (w_true, b_true), 
                       candidate_w_bs,fname=os.path.join(
                               r"C:\zUsers\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob", 
                               "weights_inferred1_prior.png"))
    
    w_draw = layer.kernel_posterior.sample()
    b_draw = layer.bias_posterior.sample()
    candidate_w_bs = []
    for _ in range(num_monte_carlo):
      w, b = sess.run((w_draw, b_draw))
      candidate_w_bs.append((w, b))
    visualize_decision(x, y, (w_true, b_true), 
                       candidate_w_bs,fname=os.path.join(
                               r"C:\zUsers\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob", 
                               "weights_inferred1_posterior.png"))
"""
