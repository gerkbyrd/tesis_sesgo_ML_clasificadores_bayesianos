# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:14:29 2019

@author: Mauricio
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:13:52 2019

@author: Mauricio
"""

from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
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

tf.reset_default_graph()
tf.compat.v1.random.set_random_seed(1234)
np.random.seed(1234)

logreg_input_matrix = np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\lgrg_in_bias.npy")
logreg_targets = np.reshape(np.load(r"C:\Users\Mauricio\Documents\Tesis\Algo Bias\Code\Tutoriales TF prob\lgrg_tgt.npy"), (6172,1))


def toy_logistic_data(num_examples, input_size=2, weights_prior_stddev=5.0):
  """Generates synthetic data for binary classification.
  Args:
    num_examples: The number of samples to generate (scalar Python `int`).
    input_size: The input space dimension (scalar Python `int`).
    weights_prior_stddev: The prior standard deviation of the weight
      vector. (scalar Python `float`).
  Returns:
    random_weights: Sampled weights as a Numpy `array` of shape
      `[input_size]`.
    random_bias: Sampled bias as a scalar Python `float`.
    design_matrix: Points sampled uniformly from the cube `[-1,
       1]^{input_size}`, as a Numpy `array` of shape `(num_examples,
       input_size)`.
    labels: Labels sampled from the logistic model `p(label=1) =
      logistic(dot(features, random_weights) + random_bias)`, as a Numpy
      `int32` `array` of shape `(num_examples, 1)`.
  """
  random_weights = weights_prior_stddev * np.random.randn(input_size)
  random_bias = np.random.randn()
  design_matrix = np.random.rand(num_examples, input_size) * 2 - 1
  logits = np.reshape(
      np.dot(design_matrix, random_weights) + random_bias,
      (-1, 1))
  p_labels = 1. / (1 + np.exp(-logits))
  labels = np.int32(p_labels > np.random.rand(num_examples, 1))
  return random_weights, random_bias, np.float32(design_matrix), labels


def plot_weights(w, b, ax, **kwargs):
    w1, w2 = w
    x1s = np.linspace(-1, 1, 100)
    x2s = -(w1  * x1s + b) / w2
    ax.plot(x1s, x2s, **kwargs)
    
    
def visualize_decision(features, labels, true_w_b, candidate_w_bs, fname):
  """Utility method to visualize decision boundaries in R^2.
  Args:
    features: Input points, as a Numpy `array` of shape `[num_examples, 2]`.
    labels: Numpy `float`-like array of shape `[num_examples, 1]` giving a
      label for each point.
    true_w_b: A `tuple` `(w, b)` where `w` is a Numpy array of
       shape `[2]` and `b` is a scalar `float`, interpreted as a
       decision rule of the form `dot(features, w) + b > 0`.
    candidate_w_bs: Python `iterable` containing tuples of the same form as
       true_w_b.
    fname: The filename to save the plot as a PNG image (Python `str`).
  """
  fig = figure.Figure(figsize=(6, 6))
  canvas = backend_agg.FigureCanvasAgg(fig)
  ax = fig.add_subplot(1, 1, 1)
  ax.scatter(features[:, 0], features[:, 1],
             c=np.float32(labels[:, 0]),
             cmap=cm.get_cmap("binary"),
             edgecolors="k")
  for w, b in candidate_w_bs:
    plot_weights(w, b, ax,
                 alpha=1./np.sqrt(len(candidate_w_bs)),
                 lw=1, color="blue")

  if true_w_b is not None:
    plot_weights(*true_w_b, ax = ax, lw=4,
                 color="green", label="true separator")

  ax.set_xlim([-1.5, 1.5])
  ax.set_ylim([-1.5, 1.5])
  ax.legend()

  canvas.print_figure(fname, format="png")
  print("saved {}".format(fname))
  
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
features, labels = build_input_pipeline(logreg_input_matrix, logreg_targets, batch_size)

# Define a logistic regression model as a Bernoulli distribution
# parameterized by logits from a single linear layer. We use the Flipout
# Monte Carlo estimator for the layer: this enables lower variance
# stochastic gradients than naive reparameterization.
with tf.compat.v1.name_scope("logistic_regression", values=[features]):
  layer = tf.layers.Dense(
          units=1, activation= None,  use_bias = False)
  logits = layer(features)
  labels_distribution = tfd.Bernoulli(logits=logits)


# Compute the -ELBO as the loss, averaged over the batch size.
neg_log_likelihood = -tf.reduce_mean(
    input_tensor=labels_distribution.log_prob(labels))
kl = sum(layer.losses) / num_examples
elbo_loss = neg_log_likelihood*1.001

# Build metrics for evaluation. Predictions are formed from a single forward
# pass of the probabilistic layers. They are cheap but noisy predictions.
predictions = tf.cast(logits > 0, dtype=tf.int32)
accuracy, accuracy_update_op = tf.compat.v1.metrics.accuracy(
    labels=labels, predictions=predictions)

with tf.compat.v1.name_scope("train"):
  optimizer = tf.compat.v1.train.AdamOptimizer(
          learning_rate = learning_rate)
  train_op = optimizer.minimize(elbo_loss)
  
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
    
    weights = layer.kernel
       
    ww = sess.run(weights)
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
