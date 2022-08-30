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
import pytest
import tempfile
import numpy as np

import tensorflow as tf
from tensorflow import keras

from boots.optimizers import GradientDescentOptimizer, LbfgsOptimizer
from boots.models import BootstrapOptimizedModel


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
    rank = 5
    model = self._build_ols_model(rank=rank)
    loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
    model.compile(loss=loss, optimizer=optimizer)
    num_samples=2**10
    np.random.seed(1)
    tf.random.set_seed(1)
    w = 1.0 + np.arange(rank)
    noise_level = 1
    X = np.random.normal(size=(num_samples, rank))
    y = np.expand_dims(X.dot(w) + noise_level * np.random.normal(size=num_samples), -1)
    model.fit(x=X, y=y, epochs=2, batch_size=2**8, shuffle=True)
    w_fit = model._tensor_converter.get_weights(model)
    w_ls, _, _, _ = np.linalg.lstsq(X, y)
    atol = np.sqrt(np.linalg.norm(w_ls - w))
    assert np.allclose(w, w_fit, atol=atol)
