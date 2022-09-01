boots
=====

The `boots` repository is an experimental implementation of mini-batch first order optimziation algorithms (Gradient Descent, L-BFGS) for optimizing stochastic loss functions in tensorflow.
We consider a loss function $f$ over a dataset ${(x_{i}, y_{i})}$ of (observation, response) pairs that is evaluated as the average of the loss $L$ evaluated for each sample:

$$ f = \frac{1}{n} \sum_{i}^{n} L(x_{i}, y_{i})$$

In our use case, the function $L$ is a measurement of goodness of fit of a predictive model $M(w)$ parameterized by parameter vector $w$.
For example, in a regression setting we may consider the least squared error in the prediction, and our $f$ and $L$ would be the following:
$$ f(w) = \frac{1}{n} \sum_{i}^{n} (M(x_{i}|w) - y_{i})^2$$

The experiment was an interest project to examine the convergence of running first order optimizers that use a `line search <https://en.wikipedia.org/wiki/Line_search>` in each iteration to minimize $f$ along a chosen search direction; that is, in each iteration to determine an approximate solution to the problem
$$ \alpha^{\*} = \arg \min_{\alpha} f(w + \alpha p)$$
where $w$ and $p$ are fixed.

 , but rather than evaluate the (deterministic[ish]) loss function by using every record in the training dataset, use mini-batches (data subsets) on the order of ``10,000`` samples to estimate the loss function with bootstrap sampling.
The bootstrap resampling process is also used to estimate the unbiased values of the batch function and gradient to perform a first order line search along estimated descent directions.

Algorithm
=========

We use a modified algorithm is 

We use the following algorithm:

0. **Inputs**:

  - An stochastic (loss) function ``f`` to evaluate over a dataset
  - A sampling ``batch_size``
  - The number of bootstrap samples ``batch_size``

1. Initialize a `BootstrappedStochasticFunction`

[Training Loss Convergence Traces](docs/_static/convnet_loss_trace.png)
