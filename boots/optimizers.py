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
      shuffle_on_precompute=False,
      max_samples=2**10,
      max_cache_entries=2**2,
      bound_bootstraps_by_samples=False,
      seed=None,
  ): 
    self.num_bootstraps = num_bootstraps
    self.subtract_bias = subtract_bias
    self.precompute = precompute
    self.shuffle_on_precompute = shuffle_on_precompute
    self.max_samples = max_samples
    self.bound_bootstraps_by_samples = bound_bootstraps_by_samples
    self.seed = seed
    self.rng = default_rng(seed=seed)
    if self.precompute:
      if max_samples is None:
        raise ValueError("Cannot precompute the sampling matrix without max_samples provided")
      self.bootstrap_sample_weights = self._build_sampling_matrix(
        num_bootstraps, max_samples, rng=self.rng
      )
    else:
      self.bootstrap_sample_weights = None
    self.cache = OptimizationStateCache(max_entries=max_cache_entries)

  def func_and_grad(self, x):
    raise NotImplementedError

  def func(self, x):
    return self.func_and_grad(x)[0]

  def grad(self, x):
    return self.func_and_grad(x)[1]

  def sample(self, num_samples):
    # Do not create more bootstrap samples than actual samples
    if self.bound_bootstraps_by_samples:
      b = np.minimum(self.num_bootstraps, num_samples)
    else:
      b = self.num_bootstraps
    if self.precompute: 
      assert self.bootstrap_sample_weights is not None
      if num_samples > self.max_samples:
        logger.warning(
          f"Bootstrap fn requested {num_samples} samples, but precomputed sampling matrix "
          f"is shape {self.bootstrap_sample_weights.shape}. Recomputing matrix."
        )
        S = self._build_sampling_matrix(num_samples, num_samples, rng=self.rng)
        self.num_bootstraps = num_samples
        self.max_samples = num_samples
        self.bootstrap_sample_weights = S
        return S
      if self.shuffle_on_precompute:
        self.rng.shuffle(self.bootstrap_sample_weights, axis=1)
      return self.bootstrap_sample_weights[:b, :num_samples]
    return self._build_sampling_matrix(b, num_samples)


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
      # Subtracting the bias increases the uncertainty in the value
      # There should be a factor of 2 is from the addition of variances
      # in quadrature ... I think lol, assuming no covariance terms...
      # v_std_bs *= np.sqrt(2)
    return (v_mean, v_std_bs, b)

  @OptimizationStateCache.cached(key='bootstrapped_f')
  def bootstrap_func(self, x):
    return self._bootstrap(x, self.func)

  @OptimizationStateCache.cached(key='bootstrapped_g')
  def bootstrap_grad(self, x):
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
    self.cache = OptimizationStateCache(max_entries=2**3)

  def minimize(self, fn, p, x0):

    c1, c2 = self.linesearch_config['c1'], self.linesearch_config['c2']
    bsfunc = fn.bootstrap_func
    bsgrad = fn.bootstrap_grad

    def bsgrad(x):
      return fn.bootstrap_grad(x)

    def _directional_product(x, dx):
      pTx = np.dot(p, x)
      pdx = np.abs(p) * dx
      dpTx = np.sqrt(np.dot(pdx, pdx))
      return pTx, dpTx

    iter_num = 0
    candidate_num = 0
    f0, df0, n0 = bsfunc(x0)
    g0, dg0, _  = bsgrad(x0)
    pTg0, dpTg0 = _directional_product(g0, dg0)

    if pTg0 > 0.0:
      logger.error("Provided direction is non-descent direction: pTg = {np.dot(p, g0)}; reversing search.")
      p = -p
      pTg0 = -pTg0

    def _sufficient_decrease_ttest(alpha, x, f, g, significance_level=0.05):
      nonlocal iter_num
      del f, g
      f_wolfe_max = f0 + c1 * alpha * pTg0
      assert f_wolfe_max <= f0, f"f_wolfe_max={f_wolfe_max} > f0={f0}"
      df_wolfe_max = np.sqrt(df0 ** 2 + (c1 * alpha * dpTg0)**2)

      f, df, n = bsfunc(x)
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
        f"f_max=({f_wolfe_max:2.4g} +/- {df_wolfe_max:2.4g}) "
        f"f0=({f0:2.4g} +/- {df0:2.4g}) with slope: pTg0={pTg0}"
      )
      return result

    def _curvature_ttest(alpha, x, f, g, significance_level=0.05):
      nonlocal iter_num
      del f, g
      pTg_wolfe_max = np.abs(c2 * pTg0)
      dpTg_wolfe_max = np.abs(c2 * dpTg0)

      g, dg, n = bsgrad(x)
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
      return bsfunc(x)[0] 

    g = lambda x: bsgrad(x)[0]

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
      fun=bsfunc(x) if success else (f0, df0, n0),
      jac=bsgrad(x) if success else (g0, dg0, n0),
      nfev=iter_num,
      njev=iter_num,
      nit=iter_num,
      maxcv=0.0,
    )
    return result


class OptimizerExitStatus(enum.IntEnum):
  RUNNING = 0
  EXITED_CONVERGENCE = 1
  EXITED_NO_CONVERGENCE = 2

  def message(self):
    return {
      self.RUNNING: "Iteration terminated successfully.",
      self.EXITED_CONVERGENCE: "Iteration terminated due to convergence criteria being met.",
      self.EXITED_NO_CONVERGENCE: "Iteration terminated abonormally without convergence.",
    }.get(self.value)


class BootstrappedFirstOrderOptimizer:

  history_ls_result_keys = [
    'fun', 'success', 'status',
  ] 
  history_keys = [
    'g_norm', 'is_steepest_descent', 'stepsize',
  ]
  
  def __init__(self, linesearch=None, convergence_window=1, convergence_frac=1.0):
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
    n_failed = 0
    for e in self.history[-self.convergence_window:]:
      n_failed += int(not e['success'])
    return (1.0 * n_failed / w) >= self.convergence_frac
    
  def compute_search_direction(self, x, f, g):
    raise NotImplementedError

  def should_restart(self, **context):
    return False

  def on_iterate_begin(self, **context):
    pass

  def on_iterate_end(self, **context):
    pass

  def is_sample_within_previous_estimate(self, f, df, index=-1):
    if len(self.history) == 0:
      return True
    previous = self.history[index]['fun']
    v1 = (previous[0] - previous[1], previous[0] + previous[1])
    v2 = (f - df, f + df)
    has_overlap = (
      (v2[0] <= v1[1]) and (v2[1] >= v1[0])
      or
      (v1[0] <= v2[1]) and (v1[1] >= v2[0])
    )
    return has_overlap

  def iterate(self, bootstrap_fn, x):
    """
    Perform the mathematical logic for a single iteration
    """
    context = {'bootstrap_fn': bootstrap_fn, 'x': x}
    self.on_iterate_begin(**context)

    # TODO: the caching mechanism is F'd here, because the
    # conversion between the tf.float32 and numpy arrays
    # changes some of the bits, such that the hashes aren't
    # the same (which is correct, but the fluctuating decimal 
    # places here are <= 10^-7 so not important in reality)
    # Not sure how to fix this right now. What's a few extra
    # fevals between friends?
    # if len(bootstrap_fn.cache.entries):
      # logger.debug("last entry: ")
      # logger.debug(bootstrap_fn.cache.entries[-1])
      # logger.debug("cache: ")
      # logger.debug(bootstrap_fn.cache.state_dict)
    logger.debug(f"Calling bootstrap func with x={x}")
    f, df, nf = bootstrap_fn.bootstrap_func(x)
    g, dg, ng = bootstrap_fn.bootstrap_grad(x)

    diff_test = self.is_sample_within_previous_estimate(f, df)
    if not diff_test:
      logger.warning(
        f"Sampled f={f:2.4g}+/-{df:2.4g} is different from sample in previous batch. "
        f"Previous: {self.history[-1]['fun'][0]:2.4g} +/- {self.history[-1]['fun'][1]:2.4g} "
        "Iteration has sampling error limits. Please increase the (bootstrap) batch_size."
      )
    # logger.debug(f"Latest cache entry: {bootstrap_fn.cache.entries[-1]}")
    p, is_steepest_descent = self.compute_search_direction(x, f, g)
    ls_result = self.linesearch.minimize(bootstrap_fn, p=p, x0=x)
    context.update({
      'f': f,
      'df': df,
      'nf': nf,
      'g': g,
      'dg': dg,
      'ng': ng,
      'g_norm': np.linalg.norm(ls_result.jac[0]),
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
    is_converged = self.is_converged() 
    while not is_converged and (k < maxiters):
      result = self.iterate(bootstrap_fn, x)
      is_converged = self.is_converged() 
      x = result.x
      k += 1
    return result


class GradientDescentOptimizer(BootstrappedFirstOrderOptimizer):

  def compute_search_direction(self, x, f, g):
    return -g, True


class LbfgsOptimizer(BootstrappedFirstOrderOptimizer):

  def __init__(self, maxcorr=2**3, **kwargs):
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
    yk = ls_result.jac[0] - context['g']
    self.s.append(sk)
    self.y.append(yk)

    if len(self.s) > self.maxcorr:
      self.s = self.s[1:]
      self.y = self.y[1:]
