import logging
import pytest
import tempfile
import numpy as np

import tensorflow as tf
from tensorflow import keras

from bootsgd.optimizers import GradientDescentOptimizer, LbfgsOptimizer
from bootsgd.models import BootstrapOptimizedModel


class TestBootstrapOptimizedModel:

  def _build_ols_model(self, rank=20):
    model_input = keras.Input(shape=(rank,), name="input")
    model_output = keras.layers.Dense(
      units=1,
      activation=None,
      use_bias=False,
      kernel_regularizer=keras.regularizers.L2(1e-9),
      kernel_initializer="ones",
    )(model_input)
    model = BootstrapOptimizedModel(
      inputs=model_input,
      outputs=model_output,
    )
    return model

  def test_string_optimizer_raises(self):
    model = self._build_ols_model()
    loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
    with pytest.raises(ValueError):
      model.compile(optimizer='string')

  def test_string_loss_raises(self):
    model = self._build_ols_model()
    optimizer = GradientDescentOptimizer()
    with pytest.raises(ValueError):
      model.compile(loss="mean_squared_error", optimizer=optimizer)

  def test_reduced_loss_raises(self):
    model = self._build_ols_model()
    optimizer = GradientDescentOptimizer()
    loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)
    with pytest.raises(ValueError):
      model.compile(loss=loss, optimizer=optimizer)

  @pytest.mark.parametrize("optimizer", [
    GradientDescentOptimizer(),
    LbfgsOptimizer(),
  ])
  def test_fit(self, optimizer):
    rank = 2
    model = self._build_ols_model(rank=rank)
    loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
    model.compile(loss=loss, optimizer=optimizer)
    num_samples=2**6
    np.random.seed(1)
    tf.random.set_seed(1)
    w = 1.0 + np.arange(rank)
    noise_level = 1
    X = np.random.normal(size=(num_samples, rank))
    y = np.expand_dims(X.dot(w) + noise_level * np.random.normal(size=num_samples), -1)
    model.fit(x=X, y=y, epochs=1, batch_size=2**4, shuffle=True)
    w_fit = model._tensor_converter.get_weights(model)
    assert np.allclose(w, w_fit, atol=0.1)
 
