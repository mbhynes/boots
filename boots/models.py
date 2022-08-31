# MIT License
#
# Copyright (c) 2022 Michael B Hynes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from copy import deepcopy
from types import MethodType

import numpy as np
import scipy

import tensorflow as tf
from tensorflow.python.platform import tf_logging
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops

from keras.utils import traceback_utils
from tensorflow.python.distribute.coordinator import cluster_coordinator
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training_utils

from tensorflow import keras
from keras import callbacks as callbacks_module

from kormos.utils.cache import OptimizationStateCache

from boots.optimizers import (
  BootstrappedDifferentiableFunction,
  BootstrappedFirstOrderOptimizer,
  GradientDescentOptimizer
)

logger = logging.getLogger(__name__)


class TensorArrayConverter:

  def __init__(self, dtype=tf.float64):
    self.dtype= dtype
    self._built = False

  def build(self, model):
    # This code has been modified from a program with the original copyright notice:
    #
    # Copyright © 2019 Pi-Yueh Chuang <pychuang@gwu.edu>
    # Distributed under terms of the MIT license.

    shapes = tf.shape_n(model.trainable_weights)
    n_tensors = len(shapes)

    idx_end = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for part_num, shape in enumerate(shapes):
      num_elements = np.product(shape)
      idx_start = idx_end
      idx_end = idx_start + num_elements
      idx.append(
          tf.reshape(tf.range(idx_start, idx_end, dtype=tf.int32), shape)
      )
      part.extend(num_elements * [part_num])

    self.num_model_variables = idx_end
    self.part = tf.constant(part)
    self.idx = idx
    self.shapes = shapes
    self.n_tensors = n_tensors
    self._built = True
    return self

  def _numpy_to_tensors(self, x):
    """
    Convert a (vector) `numpy.ndarray` of parameters to a list of tensors.

    Args:
      x (numpy.ndarray): vector of parameters used by `scipy.optimize.minimize`

    Returns:
      A list of tensors matching the shapes of this object's `model.trainable_weights`
    """
    assert self._built
    parts = tf.dynamic_partition(x, self.part, self.n_tensors)
    return [tf.reshape(part, self.shapes[k]) for (k, part) in enumerate(parts)]

  def _flatten_tensors(self, weights):
    assert self._built
    return tf.cast(tf.dynamic_stitch(self.idx, weights), dtype=self.dtype)

  def _tensors_to_numpy(self, weights):
    """
    Convert a list of tensors to a (vector) `numpy.ndarray` of parameters.

    Args:
      weights: list of tensors matching the shapes of this object's `model.trainable_weights`

    Returns:
      numpy.ndarray: vector of parameters used by `scipy.optimize.minimize`
    """
    assert self._built
    return self._flatten_tensors(weights).numpy()

  def get_weights(self, model):
    """
    Retrieve the `model` weights as a `numpy.ndarray` vector.

    Returns:
      numpy.ndarray: vector of parameters used by `scipy.optimize.minimize`
    """
    assert self._built
    return self._tensors_to_numpy(model.trainable_variables)

  def set_weights(self, x, model):
    """
    Set a `keras` model's `model.trainable_weights`.

    Args:
      x (numpy.ndarray): a parameter vector
      model (keras.Model): a model to set the weights for
    """
    assert self._built
    params = tf.dynamic_partition(
      tf.convert_to_tensor(x),
      self.part,
      self.n_tensors,
    )
    for i, (shape, param) in enumerate(zip(self.shapes, params)):
      model.trainable_variables[i].assign(
        tf.cast(tf.reshape(param, shape), model.trainable_variables[i].dtype)
      )


class BootstrapOptimizedModel(keras.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._tensor_converter = TensorArrayConverter()
    self._history = []

  def compile(self, **kwargs):
    """
    Configure the model for training.

    If the `optimizer` argument is specified as one of the `keras.optimizers.*`
    (or a string identifier thereby), then this method will simply call the
    parent method `keras.Model.compile` with the arguments provided.
    Subsequent calls to the `model.fit` will perform training using the
    standard `keras.Model.fit` method.

    If the `optimizer` argument is specified as a valid `boots.optimizers.BatchOptimizer`
    (or a valid string identifier to create one using `boots.optimizers.get()`), then
    the model will be configured to map calls to `fit` to the method `fit_batch`.

    Keyword Args:
      optimizer: A `keras.optimizer.Optimizer` or string identifier, or a `boots.optimizer.BatchOptimizer` or string identifier
      **kwargs: all other `kwargs` as passed to `keras.Model.compile <https://keras.io/api/models/model_training_apis/#compile-method>`_
    """
    jacobian_batch_size = kwargs.pop("jacobian_batch_size", 2**2)
    # Only re-use previous values at sufficiently large batch sizes
    reuse_previous_batch_fg = kwargs.pop(
      "reuse_previous_batch_fg",
      (kwargs.get('batch_size', 0) > 2**10)
    )
    orig_run_eagerly = kwargs.pop("run_eagerly", None)
    orig_optimizer = kwargs.pop("optimizer", "rmsprop")
    bootstrap_fn = kwargs.pop("bootstrap_fn", BootstrappedDifferentiableFunction()) 
    use_bootstrap_fit = False
    try:
      optimizer = keras.optimizers.get(orig_optimizer)
      run_eagerly = orig_run_eagerly
      logger.warning(f"{type(self)} compiled with optimizer={optimizer}.")
    except ValueError:
      optimizer = "rmsprop"  # Set a valid default to call .compile() as a dummy
      run_eagerly = True
      use_bootstrap_fit = True

    super().compile(optimizer=optimizer, run_eagerly=run_eagerly, **kwargs)
    # If we are fitting with a deterministic batch algorithm, reset
    # the optimizer to the desired one after compilation
    if use_bootstrap_fit:
      optimizer = orig_optimizer
      if not issubclass(type(optimizer), BootstrappedFirstOrderOptimizer):
        raise ValueError(
          f"optimizer type={type(optimizer)} was provided; please provide an instantiated BootstrappedFirstOrderOptimizer"
        )

      loss = kwargs.pop("loss")
      if loss is None:
        raise ValueError(
          f"loss=None was provided with a bootstrap optimizer: {orig_optimizer}"
        )
      if type(loss) is str:
        raise ValueError(
          f"String value for loss='{loss}' is not allowed. "
          "Please provided an instantiated loss function with reduction=keras.losses.Reduction.NONE"
        )
      if loss.reduction != keras.losses.Reduction.NONE:
        raise ValueError(
          f"Loss function {loss} has reduction {loss.reduction}. "
          "Please specify reduction=keras.losses.Reduction.NONE"
        )

      self.train_step = self.bootstrap_train_step
      self.optimizer = optimizer
      self._jacobian_batch_size = jacobian_batch_size
      self._reuse_previous_batch_fg = reuse_previous_batch_fg
      self._bootstrap_fn = bootstrap_fn
      self._tensor_converter.build(self)
    else:
      # If this model is being re-compiled, reset the fit method to the parent
      self.train_step = super().train_step

  @tf.function
  def _fg(self, X, Y):
    model = self
    batch_size = self._jacobian_batch_size
    # Chunk up the dataset into smaller subbatches since tape.jacobian blows up
    data = (
      tf.data.Dataset.from_tensor_slices(tensors=(X, Y))
      .batch(batch_size, drop_remainder=True)
    )
    # Initialize TensorArray buffers to iteratively append the loss/jacobians
    losses_array = tf.TensorArray(
      dtype=keras.backend.floatx(),
      size=len(X) // batch_size,
      element_shape=(batch_size,),
    )
    jac_array = tf.TensorArray(
      dtype=keras.backend.floatx(),
      size=len(X) // batch_size,
      element_shape=(batch_size, self._tensor_converter.num_model_variables),
    )

    for (k, record) in data.enumerate():
      x, y, _ = data_adapter.unpack_x_y_sample_weight(record)
      # tf.print("x=", x)
      # tf.print("y=", y)

      with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        losses = model.compiled_loss(y, y_pred)

        if len(model.losses):
          reg_loss = math_ops.add_n(losses_utils.cast_losses_to_common_dtype(model.losses))
          losses += tf.cast(reg_loss, dtype=losses.dtype)

      jac = tape.jacobian(losses, model.trainable_weights)
      flat_jac = tf.concat([tf.reshape(j, (x.shape[0], -1)) for j in jac], axis=-1)

      # tf.print("jac=", jac)
      # tf.print("flat_jac=", flat_jac)

      # Write the loss and jacobian arrays for this chunk to the buffer
      losses_array = losses_array.write(tf.cast(k, dtype=tf.int32), tf.transpose(losses))
      jac_array = jac_array.write(tf.cast(k, dtype=tf.int32), tf.squeeze(flat_jac))

    f = losses_array.concat()
    g = jac_array.concat()
    # tf.print("f (tf)=", f)
    # tf.print("g (tf)=", g)
    return f, g

  def bootstrap_train_step(self, data):
    nfev = 0
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    @OptimizationStateCache.cached(key='fg')
    def _wrapped_fg(_, weights):
      nonlocal nfev
      nfev += 1
      self._tensor_converter.set_weights(weights, model=self)
      f, g = self._fg(x, y)
      return f.numpy(), g.numpy()

    x0 = self._tensor_converter.get_weights(self)
    self._bootstrap_fn.func_and_grad = MethodType(_wrapped_fg, self._bootstrap_fn) 

    # In principle we should be able to re-use the last computed func/grad point
    # from the previous batch as the starting point for the current search, 
    # if the cache contains the np.array representation of the model's parameters. 
    # However there's a problem here with the single -> double precision conversion,
    # between the tensorflow tensors and the np.array values, such that the cache
    # never returns a hit between successive iterations. Caching only works with
    # tf.float46 precision.
    if not self._reuse_previous_batch_fg:
      self._bootstrap_fn.cache.clear()

    result = self.optimizer.iterate(self._bootstrap_fn, x0) 
    if result.success:
      self._tensor_converter.set_weights(result.x, self) 
    else:
      self._bootstrap_fn.cache.clear()
      if self.optimizer.is_converged():
        self.stop_training = True

    # Add function evals to the optimizer's history metrics
    # TODO: burn this, it's ugly and hacky
    self.optimizer.history[-1].update({'nfev_cached': nfev})

    logger.info(
      f" - loss: {self.optimizer.history[-1]['fun'][0]:2.4g} +/- {self.optimizer.history[-1]['fun'][1]:2.4g}"
      f" - {self.optimizer.history[-1]['nfev']} f/g evals ({nfev} cached evals)"
      f" - {self.optimizer.history[-1]['status'].message()}"
    )

    # Collect metrics to return
    return_metrics = {}
    self.compiled_metrics.update_state(y, self(x, training=False), sample_weight)
    for metric in self.metrics:
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics


class BootstrapOptimizedSequentialModel(keras.models.Sequential, BootstrapOptimizedModel):
    pass
