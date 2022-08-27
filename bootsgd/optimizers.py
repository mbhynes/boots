# import tensorflow as tf
# from tensorflow import keras
#
#
# class BootstrapLineSearchOptimizer(keras.optimizers.Optimizer):
#
#   def __init__(self):
#     super().__init__(**kwargs)

# Steps:
# - Need to customize:
#   - optimizer
#     - optimizer should store {pk, gk, Hk, etc} to compute new directions
#     - initial direction is None, should be unknown until first mini-batch gradient
#   - model.train_step 
#     - should unpack the data and provide StochasticFunction(data) to optimizer
#       (optimizer should really just have a StochasticFunction that iterates; but due to current way
#       tensorflow calls the optimizer, should allow it to provide a new function context each time
#     - This should interact with the bootstrap optimizer for the dataset
#     - (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py)
#
# BootstrappedFunction
#   - should have a method func_and_grad(params) that returns a sampled value for all N points
#   - Have N samples, B bootstraps; this creates a sampling matrix S (B x N):
#      f1, g1  f2,g2  f3,g3       fN,gN
#      ================================
#        1     2       3      ... N   
# bs 1   s11  s12     s13         s1N
#    2   
#    3  
#    ..
#    B   sB1  sB2             ... sBN
#
#   - The values should be applied to the function & gradient values to compute the boostrap estimates 
#      - E[S.dot(f)] = E[BS{f}] = 1/B * S.dot(f).sum()
#      - std[f] = std[S.dot(f)]
#   - The standard errors here should be used to compute whether 2 function values have sig. diffs

# Goal: write for numpy first, wrapping scipy.optimize.line_search

import enum
import logging

import numpy as np
from numpy.random import default_rng

import scipy

from kormos.utils.cache import OptimizationStateCache

logger = logging.getLogger(__name__)


class BootstrappedDifferentiableFunction:

  def __init__(self,
      num_bootstraps=2**10,
      subtract_bias=True,
      precompute=True,
      shuffle_on_precompute=True,
      max_samples=2**10,
      seed=None,
  ): 
    self.num_bootstraps = num_bootstraps
    self.subtract_bias = subtract_bias
    self.precompute = precompute
    self.shuffle_on_precompute = shuffle_on_precompute
    self.max_samples = max_samples
    self.rng = default_rng(seed=seed)
    if self.precompute:
      if max_samples is None:
        raise ValueError("Cannot precompute the sampling matrix without max_samples provided")
      self.bootstrap_sample_weights = self._build_sampling_matrix(
        num_bootstraps, max_samples, rng=self.rng
      )
    else:
      self.bootstrap_sample_weights = None
    self.cache = OptimizationStateCache(max_entries=32)

  def func_and_grad(self, x):
    raise NotImplementedError

  def func(self, x):
    return self.func_and_grad(x)[0]

  def grad(self, x):
    return self.func_and_grad(x)[1]

  def sample(self, num_samples):
    m = np.minimum(self.num_bootstraps, num_samples)
    if self.precompute: 
      assert self.bootstrap_sample_weights is not None
      assert num_samples <= self.max_samples

      if self.shuffle_on_precompute:
        self.rng.shuffle(self.bootstrap_sample_weights, axis=1)
      return self.bootstrap_sample_weights[:m, :num_samples]
    return self._build_sampling_matrix(m, num_samples)


  def _bootstrap(self, x, fn):
    v = fn(x)
    v_mean = np.mean(v, axis=0)

    n = len(v)
    S = self.sample(n)
    (b, _) = S.shape

    # Compute the bootstrap estimates
    v_weighted = np.dot(S, v) / n
    v_std_bs = np.std(v_weighted, axis=0, ddof=1)
    if self.subtract_bias:
      v_mean_bs = np.mean(v_weighted, axis=0)
      bias = (v_mean_bs - v_mean)
      v_mean -= bias
    return (v_mean, v_std_bs, b)

  def bootstrap_func(self, x, **kwargs):
    return self._bootstrap(x, self.func)

  def bootstrap_grad(self, x, **kwargs):
    return self._bootstrap(x, self.grad)

  @staticmethod
  def _build_sampling_matrix(b, n, rng=None):
    rng = rng or default_rng(seed=None)
    shape = (b, n)
    S = np.zeros(shape=shape)
    indices = rng.integers(low=0, high=n, size=shape)
    for row in range(b):
      idx, counts = np.unique(indices[row], return_counts=True)
      S[row, idx] = counts
    return S


class LinesearchExitStatus(enum.IntEnum):
  SUCCESS = 0
  NO_ITERATION_PERFORMED = 1
  NO_WOLFE_POINTS_FOUND = 2
  STEP_UNRESOLVEABLE_WITHIN_SIG_LEVEL = 3
  NUMERICAL_ERRORS_ENCOUNTERED = 5

  def message(self):
    return {
      self.SUCCESS: "Line search terminated successfully.",
      self.NO_ITERATION_PERFORMED: "No iterations were performed by scipy.optimize.line_search.",
      self.NO_WOLFE_POINTS_FOUND: "No points satisfying the Wolfe conditions were found; try increasing maxiter.",
      self.STEP_UNRESOLVEABLE_WITHIN_SIG_LEVEL: "Points satisfying the Wolfe conditions were found but not resolveable within the significance level.",
      self.NUMERICAL_ERRORS_ENCOUNTERED: "Encountered numerical errors during computations that produced NaN values.",
    }.get(self.value)


class BootstrappedWolfeLineSearch:

  DEFAULT_LINESEARCH_CONFIG = {
    'c1': 1e-4, 
    'c2': 0.9,
    'maxiter': 20,
  }
  
  def __init__(self, linesearch_config=None, significance_level=0.05):
    self.significance_level = significance_level
    self.linesearch_config = {**self.DEFAULT_LINESEARCH_CONFIG, **(linesearch_config or {})}

  def minimize(self, fn, p, x0):

    c1, c2 = self.linesearch_config['c1'], self.linesearch_config['c2']

    def _directional_product(x, dx):
      pTx = np.dot(p, x)
      pdx = np.abs(p) * dx
      dpTx = np.sqrt(np.dot(pdx, pdx))
      return pTx, dpTx

    iter_num = 0
    candidate_num = 0
    f0, df0, n0 = fn.bootstrap_func(x0)
    g0, dg0, _  = fn.bootstrap_grad(x0)
    pTg0, dpTg0 = _directional_product(g0, dg0)

    if np.dot(p, g0) > 0.0:
      logger.error("Provided direction is non-descent direction: pTg = {np.dot(p, g0)}")
      return None

    def _sufficient_decrease_ttest(alpha, x, f, g, significance_level=0.05):
      nonlocal iter_num
      del f, g
      f_wolfe_max = f0 + c1 * alpha * pTg0
      df_wolfe_max = np.sqrt(df0 ** 2 + (c1 * alpha * dpTg0)**2)

      f, df, n = fn.bootstrap_func(x)
      statistic, pvalue = scipy.stats.ttest_ind_from_stats(
        mean1=f_wolfe_max,
        std1=df_wolfe_max,
        nobs1=n0,
        mean2=f,
        std2=df,
        nobs2=n,
        equal_var=False,
        alternative='greater', # Alternate hypothesis: mean1 > mean2
      )
      result = (pvalue <= significance_level) 
      msg = 'Accepted' if result else 'Rejected'
      logger.debug(
        f"Iter {iter_num}: (Decrease) {msg} step={alpha:2.2f} at sig={pvalue:2.2g}: "
        f"f=({f:2.4g} +/- {df:2.4g}) vs "
        f"f_max=({f_wolfe_max:2.4g} +/- {df_wolfe_max:2.4g})"
      )
      return result

    def _curvature_ttest(alpha, x, f, g, significance_level=0.05):
      nonlocal iter_num
      del f, g
      pTg_wolfe_max = np.abs(c2 * pTg0)
      dpTg_wolfe_max = np.abs(c2 * dpTg0)

      g, dg, n = fn.bootstrap_grad(x)
      pTg, dpTg = _directional_product(g, dg)
      statistic, pvalue = scipy.stats.ttest_ind_from_stats(
        mean1=pTg_wolfe_max,
        std1=dpTg_wolfe_max,
        nobs1=n0,
        mean2=pTg,
        std2=dpTg,
        nobs2=n,
        equal_var=False,
        alternative='greater', # Alternate hypothesis: mean1 > mean2
      )
      result = (pvalue <= significance_level) 
      msg = 'Accepted' if result else 'Rejected'
      logger.debug(
        f"Iter {iter_num}: (Curvature) {msg} step={alpha:2.2g} at sig={pvalue:2.2g}: "
        f"pTg=({pTg:2.2g} +/- {dpTg:2.2g}) vs "
        f"pTg_max=({pTg_wolfe_max:2.2g} +/- {dpTg_wolfe_max:2.2g})"
      )
      return result

    def _wolfe_conditional_ttest(*args):
      nonlocal candidate_num
      candidate_num += 1
      return (
        _sufficient_decrease_ttest(*args, significance_level=self.significance_level)
        and _curvature_ttest(*args, significance_level=self.significance_level)
      )

    def _determine_exit_status(alpha):
      nonlocal iter_num, candidate_num
      success = (alpha is not None) and np.isfinite(alpha)
      if success:
        status = LinesearchExitStatus.SUCCESS
      else:
        if (alpha is not None) and not np.isfinite(alpha):
          status = LinesearchExitStatus.NUMERICAL_ERRORS_ENCOUNTERED
        if iter_num == 0:
          status = LinesearchExitStatus.NO_ITERATION_PERFORMED
        elif candidate_num == 0:
          status = LinesearchExitStatus.NO_WOLFE_POINTS_FOUND
        else:
          status = LinesearchExitStatus.STEP_UNRESOLVEABLE_WITHIN_SIG_LEVEL
      return success, status, status.message()

    def f(x):
      nonlocal iter_num
      iter_num += 1
      return fn.bootstrap_func(x)[0] 

    g = lambda x: fn.bootstrap_grad(x)[0]

    (alpha, nfev, njev, new_fval, old_fval, new_slope) = scipy.optimize.line_search(
      f=f,
      myfprime=g,
      xk=x0,
      pk=p,
      extra_condition=_wolfe_conditional_ttest,
      **self.linesearch_config
    )
    success, status, msg = _determine_exit_status(alpha)
    x = x0 + alpha * p if success else x0
    result = scipy.optimize.OptimizeResult(
      x=x,
      success=success,
      status=status,
      msg=msg,
      fun=fn.bootstrap_func(x)[0],
      jac=fn.bootstrap_grad(x)[0],
      nfev=nfev,
      njev=njev,
      nit=iter_num,
      maxcv=0.0,
    )
    return result


class BootstrappedFirstOrderOptimizer:

  history_ls_result_keys = [
    'success', 'status',
  ] 
  history_keys = [
    'f', 'df', 'g_norm', 'is_steepest_descent', 'stepsize',
  ]
  
  def __init__(self, linesearch=None, convergence_window=2**3, convergence_frac=0.75):
    self.linesearch = linesearch or BootstrappedWolfeLineSearch()
    self.convergence_window = convergence_window
    self.convergence_frac = convergence_frac
    self.history = []

  def update_history(self, **context):
    entry = {
      key: val for (key, val) in context.items()
      if key in self.history_keys
    }
    ls_entry = {
      key: getattr(context['ls_result'], key)
      for key in self.history_ls_result_keys
    }
    entry.update(ls_entry)
    self.history.append(entry)

  def is_converged(self):
    w = min(len(self.history), self.convergence_window)
    if w == 0:
      return False
    n = 0
    n_failed = 0
    for e in self.history[-self.convergence_window:]:
      n += 1
      n_failed += int(not e['success'])
    return (1.0 * n_failed / n) >= self.convergence_frac
    
  def compute_search_direction(self, x, f, g):
    raise NotImplementedError

  def should_restart(self, **context):
    return False

  def on_iterate_begin(self, **context):
    pass

  def on_iterate_end(self, **context):
    pass

  def iterate(self, bootstrap_fn, x):
    """
    Perform the mathematical logic for a single iteration
    """
    context = {'bootstrap_fn': bootstrap_fn, 'x': x}
    self.on_iterate_begin(**context)

    f, df, nf = bootstrap_fn.bootstrap_func(x)
    g, dg, ng = bootstrap_fn.bootstrap_grad(x)
    p, is_steepest_descent = self.compute_search_direction(x, f, g)

    ls_result = self.linesearch.minimize(bootstrap_fn, p=p, x0=x)
    context.update({
      'f': f,
      'df': df / np.sqrt(nf),
      'g': g,
      'g_norm': np.linalg.norm(g),
      'p': p,
      'ls_result': ls_result,
      'is_steepest_descent': is_steepest_descent,
    })

    if self.should_restart(**context):
      logger.warning(f"{type(self)} triggered a restart.")
      p = -g
      ls_result = self.linesearch.minimize(bootstrap_fn, p=p, x0=x)
      context.update({'ls_result': ls_result, 'p': p, 'is_steepest_descent': True})

    # Compute the final stepsize as the norm of the difference betwen successive iterations
    context.update({'stepsize': 0.0 if not ls_result.success else np.linalg.norm(ls_result.x - x)})

    self.update_history(**context)
    self.on_iterate_end(**context)
    logger.info(self.history[-1])
    return ls_result

  def minimize(self, bootstrap_fn, x0, maxiters=20):
    k = 0
    x = x0
    while not self.is_converged() and (k < maxiters):
      result = self.iterate(bootstrap_fn, x)
      x = result.x
      k += 1
    return result


class GradientDescentOptimizer(BootstrappedFirstOrderOptimizer):

  def compute_search_direction(self, x, f, g):
    return -g, True


class LbfgsOptimizer(BootstrappedFirstOrderOptimizer):

  def __init__(self, maxcorr=10, **kwargs):
    super().__init__(**kwargs)
    self.maxcorr = maxcorr
    self.s = []
    self.y = []

  def compute_search_direction(self, x, f, g):
    if len(self.s) == 0:
      p = -g
      is_steepest_descent = True
    else:
      Hinv = scipy.optimize.LbfgsInvHessProduct(
        np.array(self.s),
        np.array(self.y),
      )
      p = -Hinv._matvec(g)
      is_steepest_descent = False

      if np.any(~np.isfinite(p)):
        logger.warning(
          "Numerical error computing L-BFGS direction; restarting with -g."
        )
        p = -g
        is_steepest_descent = True
    return p, is_steepest_descent

  def should_restart(self, **context):
    return (
      not context['is_steepest_descent']
      and not context['ls_result'].success
    )

  def on_iterate_end(self, **context):
    if context['is_steepest_descent']:
      # reset the history
      self.s, self.y = [], []

    ls_result = context['ls_result']

    if not ls_result.success:
      return

    sk = ls_result.x - context['x']
    yk = ls_result.jac - context['g']
    self.s.append(sk)
    self.y.append(yk)

    if len(self.s) > self.maxcorr:
      self.s = self.s[1:]
      self.y = self.y[1:]
