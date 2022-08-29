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

from bootsgd.optimizers import (
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

    If the `optimizer` argument is specified as a valid `kormos.optimizers.BatchOptimizer`
    (or a valid string identifier to create one using `kormos.optimizers.get()`), then
    the model will be configured to map calls to `fit` to the method `fit_batch`.

    Keyword Args:
      optimizer: A `keras.optimizer.Optimizer` or string identifier, or a `kormos.optimizer.BatchOptimizer` or string identifier
      **kwargs: all other `kwargs` as passed to `keras.Model.compile <https://keras.io/api/models/model_training_apis/#compile-method>`_
    """
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
          f"optimizer={optimizer} was provided; please provide an instantiated BootstrappedFirstOrderOptimizer"
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
      self._bootstrap_fn = bootstrap_fn
      self._tensor_converter.build(self)
    else:
      # If this model is being re-compiled, reset the fit method to the parent
      self.train_step = super().train_step

  # @tf.function
  def _fg(self, x, y, sample_weight):
    model = self
    # data = data_adapter.expand_1d(data)
    # x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
    # print("x=", x)
    # print("y=", y)
    with tf.GradientTape() as tape:
      # weights = tf.concat([tf.reshape(w, [-1]) for w in self.trainable_weights], axis=-1)
      y_pred = model(x, training=True)
      # print("y_pred=", y_pred)
      losses = model.compiled_loss(y, y_pred, sample_weight=sample_weight)

      if len(self.losses):
        reg_loss = losses_utils.cast_losses_to_common_dtype(self.losses)
        reg_loss_vec = math_ops.add_n(reg_loss) * tf.ones(shape=losses.shape, dtype=losses.dtype)
        losses += reg_loss_vec
    # weights = self._tensor_converter._flatten_tensors(self.trainable_weights)
    # print("weights=", weights)
    jac = tape.jacobian(losses, self.trainable_weights)
    # return losses, jac[0]
    # print("jac=", jac[0])
    flat_jac = tf.concat([tf.reshape(j, (x.shape[0], -1)) for j in jac], axis=-1)
    # print("flat_jac=", flat_jac)
    # jac = tape.jacobian(losses, weights)
    # print("jac=", jac)
    return losses, flat_jac

  def bootstrap_train_step(self, data):

    nfev = 0
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    @OptimizationStateCache.cached(key='fg')
    def _wrapped_fg(_, weights):
      nonlocal nfev
      nfev += 1
      self._tensor_converter.set_weights(weights, model=self)
      f, g = self._fg(x, y, sample_weight)
      # g = self._tensor_converter._tensors_to_numpy(g)
      # if g.shape[-1] == 1:
      #   g = np.squeeze(g.numpy(), axis=-1)
      # else:
      #   g = g.numpy()
      return f.numpy(), g.numpy()

    x0 = self._tensor_converter.get_weights(self)
    self._bootstrap_fn.func_and_grad = MethodType(_wrapped_fg, self._bootstrap_fn) 
    # self._bootstrap_fn.cache.clear()
    result = self.optimizer.iterate(self._bootstrap_fn, x0) 
    if result.success:
      self._tensor_converter.set_weights(result.x, self) 
    else:
      self.stop_training = True

    # Add function evals to the optimizer's history metrics
    # TODO: burn this, it's ugly and hacky
    self.optimizer.history[-1].update({'nfev': nfev})

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
