import sys
import copy
import logging
import pickle

import numpy as np
from tensorflow import keras

from bootsgd.models import BootstrapOptimizedSequentialModel
from bootsgd.optimizers import (
  BootstrappedDifferentiableFunction,
  BootstrappedWolfeLineSearch,
  BootstrappedFirstOrderOptimizer,
  GradientDescentOptimizer,
  LbfgsOptimizer
)

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)


def build_model():
  model = BootstrapOptimizedSequentialModel([
    keras.Input(shape=input_shape),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
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

def run_tests(training_data, validation_data, num_trials=1, boostrap_args=None, ls_args=None, fit_args=None):
  batch_size = fit_args.pop("batch_size", 32)

  # Update the batch size if provided
  fn_args = copy.deepcopy(boostrap_args) or {}
  fn_args.update({
    # We only need to have batch_size samples pre-computed
    "max_samples": batch_size,
    # If the data is shuffled at each epoch, don't need to shuffle the bootstrap sample weight matrix
    "shuffle_on_precompute": not fit_args.get("shuffle", False),
  })

  bootstrap_fn = BootstrappedDifferentiableFunction(**fn_args)
  linesearch = BootstrappedWolfeLineSearch(**(ls_args or {}))
  configs = {
    # "adam": {
    #   "loss": keras.losses.CategoricalCrossentropy(),
    #   "optimizer": "adam",
    # },
    # "rmsprop": {
    #   "loss": keras.losses.CategoricalCrossentropy(),
    #   "optimizer": "rmsprop",
    # },
    # "boot-sgd": {
    #   "loss": keras.losses.CategoricalCrossentropy(reduction="none"),
    #   "optimizer": GradientDescentOptimizer(linesearch=linesearch),
    #   "bootstrap_fn": bootstrap_fn,
    # },
    "boot-lbfgs": {
      "loss": keras.losses.CategoricalCrossentropy(reduction="none"),
      "optimizer": LbfgsOptimizer(linesearch=linesearch),
      "bootstrap_fn": bootstrap_fn,
    }
  }
  history = {key: [] for key in configs.keys()}

  for (name, config) in configs.items():
    for k in range(num_trials):
      model = build_model()
      model.compile(**config)
      hist = model.fit(
        x=training_data[0],
        y=training_data[1],
        validation_data=validation_data,
        **fit_args,
      )
      if issubclass(type(config.get('optimizer')), BootstrappedFirstOrderOptimizer):
        history[name].append((hist, model.optimizer.history))
      else:
        history[name].append((hist, []))
  return history

def main(args):
  training_data, validation_data = load_data()
  history = run_tests(
    training_data,
    validation_data,
    fit_args={"shuffle": True}
  )
  with open("history.pkl", "w") as f:
    pickle.dump(history, f)
  return 0

if __name__ == "__main__":
  sys.exit(main(args=sys.argv[1:]))
