boots
=====

`boots` is an experimental implementation of mini-batch first order optimziation algorithms (Gradient Descent, L-BFGS) for optimizing stochastic differentiable functions.
The functions are evaluating in batches (subsets of a dataset) on the order of ``2^10`` samples per batch, and bootstrap resampling is applied to estimate the unbiased values of the batch function and gradient to perform a first order line search along estimated descent directions.
