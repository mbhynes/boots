import logging
import random
import numpy as np

from kormos.utils.cache import OptimizationStateCache

from bootsgd.optimizers import (
  BootstrappedDifferentiableFunction,
  BootstrappedWolfeLineSearch,
  BootstrappedFirstOrderOptimizer,
  GradientDescentOptimizer,
  LbfgsOptimizer
)

logger = logging.getLogger(__name__)

def biased_resample_array(x, weights, seed=None):
  if seed is not None:
    random.seed(seed)
  idx = random.choices(
    range(len(x)),
    weights=weights,
    k=len(x),
  )
  return x[idx]


class Polynomial(BootstrappedDifferentiableFunction):

  def __init__(self, degree=2, noise=0.5, num_samples=100, **kwargs):
    super().__init__(**kwargs)
    # Store the actual values of the polynomial
    self.noise = noise
    self.w_true = 1 + np.arange(degree + 1)[::-1]
    np.random.seed(kwargs.get('seed', 1))
    self.X = np.vander(np.random.normal(size=num_samples), N=(degree + 1), increasing=True)
    self._compute_y_true()
  
  def _compute_y_true(self, x=None):
    x = x if x is not None else self.X
    np.random.seed(self.seed)
    self.y_true = np.dot(x, self.w_true) + self.noise * np.random.normal(size=len(x))

  @OptimizationStateCache.cached(key='fg')
  def func_and_grad(self, w):
    logger.debug(f"Called with w={w}")
    n = len(w)
    y_pred = np.dot(self.X[:, :n], w)
    f = (self.y_true - y_pred) ** 2
    grad = 2 * (self.X[:, :n].T * (y_pred - self.y_true)).T
    return f, grad


class TestBootstrappedDifferentiableFunction:

  def test_bootstrap_func(self):
    np.random.seed(1)
    poly = Polynomial(num_samples=1000, num_bootstraps=1000, noise=0)
    f, df, n = poly.bootstrap_func(poly.w_true)
    assert np.allclose([f, df], [0, 0])

    f, df, n = poly.bootstrap_func(0 * poly.w_true)
    expected = np.mean(poly.y_true ** 2)
    assert np.allclose(f, expected, atol=0.5)


class TestBootstrappedWolfeLineSearch:

  def _run_gradient_descent(self, w0, polyargs, is_biased=False, iters=20):
    w = w0
    ls = BootstrappedWolfeLineSearch()
    for k in range(iters):
      # Build a new polynomial, which triggers new samples to be created
      # This is a hacky simulation of 1 polynomial class subsampling a larger dataset
      poly = Polynomial(**{**polyargs, **{'seed': k}})
      if is_biased:
        poly.X = biased_resample_array(
          poly.X,
          weights=(poly.X[:, 1] - poly.X[:, 1].min()),
          seed=k,
        )
        poly._compute_y_true()
      f, g = poly.func_and_grad(w)
      p = -np.mean(g, axis=0)
      result = ls.minimize(poly, p=p, x0=w)
      logger.info(f"{k}: {result.fun}: {result.x}: {result.msg}")
      if not result.success:
        break
      w = result.x
    return w, poly

  def test_exact_line_search(self):
    np.random.seed(1)
    degree = 2
    w0 = np.random.normal(size=(degree + 1))
    polyargs = dict(num_bootstraps=2**8, degree=degree, noise=0, subtract_bias=False)
    w, poly = self._run_gradient_descent(w0, polyargs=polyargs, is_biased=False)
    assert np.allclose(w, poly.w_true, atol=1e-2)

  def test_biased_exact_line_search(self):
    np.random.seed(1)
    degree = 2
    w0 = np.random.normal(size=(degree + 1))
    polyargs = dict(num_samples=2**10, num_bootstraps=2**10, degree=degree, noise=0, subtract_bias=True, seed=1)
    w, poly = self._run_gradient_descent(w0, polyargs=polyargs, is_biased=True)
    assert np.allclose(w, poly.w_true, atol=1e-2)

  def test_biased_inexact_line_search(self):
    degree = 5
    polyargs = dict(num_samples=2**10, num_bootstraps=2**10, degree=degree, noise=1, subtract_bias=True, seed=1)
    np.random.seed(1)
    w0 = np.random.normal(size=(degree + 1))
    w, poly = self._run_gradient_descent(w0, polyargs=polyargs, is_biased=True, iters=10)
    # Solve the system by linear least squares to get a ballpark solution tolerance
    w_ls, _, _, _ = np.linalg.lstsq(poly.X, poly.y_true)
    atol = np.sqrt(np.linalg.norm(w_ls - poly.w_true))
    assert np.allclose(w, poly.w_true, atol=atol)


class OptimizerTest:

  optimizer_cls = BootstrappedFirstOrderOptimizer

  def _iterate(self, w0, polyargs, is_biased=True, iters=20):
    w = w0
    opt = self.optimizer_cls(linesearch=BootstrappedWolfeLineSearch())
    
    for k in range(iters):
      # Build a new polynomial, which triggers new samples to be created
      # This is a hacky simulation of 1 polynomial class subsampling a larger dataset
      poly = Polynomial(**{**polyargs, **{'seed': k}})
      if is_biased:
        poly.X = biased_resample_array(poly.X, weights=(poly.X[:, 1] - poly.X[:, 1].min()), seed=k)
        poly._compute_y_true()
      result = opt.iterate(poly, x=w)
      logger.info(f"{k}: {result.fun}: {result.x}: {result.msg}")
      if not result.success:
        break
      w = result.x
    return w, poly


class TestGradientDescentOptimizer(OptimizerTest):

  optimizer_cls = GradientDescentOptimizer

  def test_minimize(self):
    np.random.seed(1)
    degree = 2
    w0 = np.random.normal(size=(degree + 1))
    poly = Polynomial(
      num_samples=2**10,
      num_bootstraps=2**10,
      degree=degree,
      noise=1,
      subtract_bias=True,
      seed=1,
    )
    opt = self.optimizer_cls(linesearch=BootstrappedWolfeLineSearch())
    result = opt.minimize(poly, w0, maxiters=20 * degree)
    # Solve the system by linear least squares to get a ballpark solution tolerance
    w_ls, _, _, _ = np.linalg.lstsq(poly.X, poly.y_true)
    atol = np.sqrt(np.linalg.norm(w_ls - poly.w_true))
    assert np.allclose(result.x, poly.w_true, atol=atol)

  def test_iterate(self):
    np.random.seed(1)
    degree = 2
    w0 = np.random.normal(size=(degree + 1))
    polyargs = dict(
      num_samples=2**8,
      num_bootstraps=2**8,
      degree=degree,
      noise=1,
      subtract_bias=True,
      seed=1,
    )
    w, poly = self._iterate(w0, polyargs, is_biased=True, iters=10)
    # Solve the system by linear least squares to get a ballpark solution tolerance
    w_ls, _, _, _ = np.linalg.lstsq(poly.X, poly.y_true)
    atol = np.sqrt(np.linalg.norm(w_ls - poly.w_true))
    assert np.allclose(w, poly.w_true, atol=atol)


class TestLbfgsOptimizer(OptimizerTest):
  
  optimizer_cls = LbfgsOptimizer

  def test_minimize(self):
    np.random.seed(1)
    degree = 5
    w0 = np.random.normal(size=(degree + 1))
    poly = Polynomial(
      num_samples=2**10,
      num_bootstraps=2**10,
      degree=degree,
      noise=1,
      subtract_bias=True,
    )
    opt = self.optimizer_cls(linesearch=BootstrappedWolfeLineSearch())
    result = opt.minimize(poly, w0, maxiters=3*degree)
    # Solve the system by linear least squares to get a ballpark solution tolerance
    w_ls, _, _, _ = np.linalg.lstsq(poly.X, poly.y_true)
    atol = np.sqrt(np.linalg.norm(w_ls - poly.w_true))
    assert np.allclose(result.x, poly.w_true, atol=atol)

  def test_iterate(self):
    np.random.seed(1)
    degree = 5
    w0 = np.random.normal(size=(degree + 1))
    polyargs = dict(
      num_samples=2**10,
      num_bootstraps=2**10,
      degree=degree,
      noise=1,
      subtract_bias=True,
    )
    w, poly = self._iterate(w0, polyargs, is_biased=True, iters=3*degree)
    # Solve the system by linear least squares to get a ballpark solution tolerance
    w_ls, _, _, _ = np.linalg.lstsq(poly.X, poly.y_true)
    atol = np.sqrt(np.linalg.norm(w_ls - poly.w_true))
    assert np.allclose(w, poly.w_true, atol=atol)
