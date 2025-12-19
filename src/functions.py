"""
Objective functions and their derivatives for optimization problems.

This module implements various test functions for convex optimization,
including their gradients and Hessians.
"""

import numpy as np


def f0(x):
    """
    Q1(b): f0(x1, x2) = log(e^x1 + e^x2) + 0.5*(x1^2 + x2^2)

    Uses stable log-sum-exp computation to avoid numerical overflow.

    Parameters
    ----------
    x : array_like, shape (2,)
        Input point [x1, x2].

    Returns
    -------
    float
        Function value at x.
    """
    # Stable log-sum-exp: log(e^x1 + e^x2) = max(x1,x2) + log(e^(x1-max) + e^(x2-max))
    x_max = np.max(x)
    log_sum_exp = x_max + np.log(np.exp(x[0] - x_max) + np.exp(x[1] - x_max))
    return log_sum_exp + 0.5 * (x[0]**2 + x[1]**2)


def grad_f0(x):
    """
    Gradient of f0 using softmax.

    The gradient is given by:
        grad_f0 = [p1 + x1, p2 + x2]
    where [p1, p2] = softmax([x1, x2])

    Parameters
    ----------
    x : array_like, shape (2,)
        Input point [x1, x2].

    Returns
    -------
    ndarray, shape (2,)
        Gradient vector at x.
    """
    # Stable softmax computation
    x_max = np.max(x)
    exp_shifted = np.exp(x - x_max)
    softmax = exp_shifted / np.sum(exp_shifted)

    return softmax + x


def hess_f0(x):
    """
    Hessian of f0 using softmax.

    The Hessian is given by:
        H = diag(p1, p2) - [p1*p2, p1*p2; p1*p2, p1*p2] + I
    where [p1, p2] = softmax([x1, x2])

    This can be written as:
        H = diag(softmax) - outer(softmax, softmax) + I

    Parameters
    ----------
    x : array_like, shape (2,)
        Input point [x1, x2].

    Returns
    -------
    ndarray, shape (2, 2)
        Hessian matrix at x.
    """
    # Stable softmax computation
    x_max = np.max(x)
    exp_shifted = np.exp(x - x_max)
    softmax = exp_shifted / np.sum(exp_shifted)

    # Hessian: diag(softmax) - outer(softmax, softmax) + I
    H = np.diag(softmax) - np.outer(softmax, softmax) + np.eye(2)

    return H
