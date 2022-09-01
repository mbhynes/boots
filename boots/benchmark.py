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

import copy
import logging
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
keras.backend.set_floatx("float64")

from boots.models import BootstrapOptimizedSequentialModel
from boots.optimizers import (
  BootstrappedDifferentiableFunction,
  BootstrappedWolfeLineSearch,
  BootstrappedFirstOrderOptimizer,
  GradientDescentOptimizer,
  LbfgsOptimizer
)

logging.basicConfig(level=logging.INFO)

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

def build_model():
  """
  Create a simple convolutional neural net for MNIST digit classification.
  This model is adapted from: https://keras.io/examples/vision/mnist_convnet/
  """
  model = BootstrapOptimizedSequentialModel([
    keras.Input(shape=input_shape),
    keras.layers.Conv2D(16, kernel_size=(4, 4), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(5, 5)),
    keras.layers.Flatten(),
    keras.layers.Dense(num_classes, activation="softmax"),
  ])
  return model

def load_data():
  # Load the data and split it between train and test sets
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

  # Scale images to the [0, 1] range
  x_train = x_train.astype("float32") / 255
  x_test = x_test.astype("float32") / 255
  # Make sure images have shape (28, 28, 1)
  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)
  print("x_train shape:", x_train.shape)
  print(x_train.shape[0], "train samples")
  print(x_test.shape[0], "test samples")

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return (x_train, y_train), (x_test, y_test)

def run_tests(training_data=None, validation_data=None, num_trials=1, fn_args=None, ls_args=None, fit_args=None):
  """
  Run a suite of simple .fit() optimization tests on the convnet model.
  """

  if training_data is None:
    training_data, validation_data = load_data()

  fit_args = fit_args or {}

  bootstrap_fn = BootstrappedDifferentiableFunction(**(fn_args or {}))
  linesearch = BootstrappedWolfeLineSearch(**(ls_args or {}))
  configs = {
    "adam": {
      "compile_args": {
        "loss": keras.losses.CategoricalCrossentropy(),
        "optimizer": "adam",
        "metrics": ["accuracy"],
      },
      "fit_args": {
        "steps_per_epoch": 2**7,
        "batch_size": 2**5,
        # Run for ~3 epochs since each bootstrapped l-bfgs iter will take ~3 function evals
        "epochs": 3 * len(training_data[0]) // (2**5 * 2**7),
      },
    },
    "boot-sgd": {
      "compile_args": {
        "loss": keras.losses.CategoricalCrossentropy(reduction="none"),
        "optimizer": GradientDescentOptimizer(linesearch=linesearch, convergence_window=8),
        "bootstrap_fn": bootstrap_fn,
        "metrics": ["accuracy"],
        "jacobian_batch_size": 2,
        "reuse_previous_batch_fg": False,
      },
      "fit_args": {
        'batch_size': 2**8,
        'epochs': 1,
      },
    },
    "boot-lbfgs": {
      "compile_args": {
        "loss": keras.losses.CategoricalCrossentropy(reduction="none"),
        "optimizer": LbfgsOptimizer(linesearch=linesearch, convergence_window=8),
        "bootstrap_fn": bootstrap_fn,
        "metrics": ["accuracy"],
        "jacobian_batch_size": 2,
        "reuse_previous_batch_fg": False,
      },
      "fit_args": {
        'batch_size': 2**8,
        'epochs': 1,
      },
    }
  }
  models = {key: [] for key in configs.keys()}
  history = {key: [] for key in configs.keys()}

  for (name, config) in configs.items():
    for k in range(num_trials):
      model = build_model()
      model.compile(**config["compile_args"])
      hist = model.fit(
        x=training_data[0],
        y=training_data[1],
        validation_data=validation_data,
        **{**fit_args, **config.get("fit_args", {})},
      )
      if issubclass(type(config.get("compile_args", {}).get("optimizer")), BootstrappedFirstOrderOptimizer):
        history[name].append((hist, model.optimizer.history))
      else:
        history[name].append((hist, []))
      models[name].append(model)
  return models, history, configs

def plot_results(models, history, steps_per_epoch=1, batch_size=2**5, trialno=0):
  keys = models.keys()
  ax = plt.gca()
  for key in keys:
    # bit lazy, should average/quantile the trials but just take 1 trial for now
    model = models[key][trialno]
    hist = history[key][trialno]
    if issubclass(type(model.optimizer), BootstrappedFirstOrderOptimizer):
      df = pd.DataFrame(hist[1])
      df['f'] = df['fun'].apply(lambda x: x[0])
      df['df'] = df['fun'].apply(lambda x: x[1])
      df['num_datapoints'] = df['fun'].apply(lambda x: x[-1])
      df['n'] = df['num_datapoints'] * df['nfev_cached']
      df['val_loss'] = np.nan
      df['n_cum'] = df['n'].cumsum()
      p = plt.plot(df['n_cum'], df['f'], label=key)
      plt.fill_between(
        df['n_cum'],
        (df['f'] - df['df']).ewm(halflife=1).mean(),
        (df['f'] + df['df']).ewm(halflife=1).mean(),
        alpha=0.15,
        color=p[0].get_color(),
      )
    else:
      df = pd.DataFrame(hist[0].history)
      df['f'] = df['loss']
      df['n'] = steps_per_epoch * batch_size
      df['n_cum'] = df['n'].cumsum()
      plt.plot(df['n_cum'], df['f'], label=key)

  plt.legend()
  plt.xlabel(f"# Function/Gradient Evaluations")
  plt.ylabel("Loss Function Value")
  return ax


def main(args):
  try:
    training_data, validation_data = load_data()
    models, history, configs = run_tests(
      training_data,
      validation_data,
      ls_args={
        'linesearch_config': {'maxiter': 5},
        'significance_level': 0.25,
      },
      fn_args={
        'num_bootstraps': 2**9
      },
    )
    ax = plot_results(models, history, steps_per_epoch=configs['adam']['fit_args']['steps_per_epoch'])
    plt.savefig("docs/_static/convnet_loss_trace.png")
    with open("docs/_static/convnet_history.pkl", "w") as f:
      pickle.dump(history, f)
    return 0
  except Exception:
    return 1


if __name__ == "__main__":
  sys.exit(main(args=sys.argv[1:]))
