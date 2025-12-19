"""
Optimization algorithms for unconstrained minimization.

This module implements Newton's method with backtracking line search.
"""

import numpy as np


def line_search(x, delta_x, f, grad_f, alpha=0.1, beta=0.5, max_it=25):
    """
    Backtracking line search.

    Parameters
    ----------
    x : array_like, shape (n,)
        Current point.
    delta_x : array_like, shape (n,)
        Descent direction.
    f : callable
        Objective function; f(x) -> scalar.
    grad_f : callable
        Gradient; grad_f(x) -> array_like shape (n,).
    alpha : float,
        Sufficient decrease parameter (0 < alpha < 0.5).
    beta : float,
        Shrinkage factor (0 < beta < 1).
    max_it : int,
        Maximum number of backtracking steps.

    Returns
    -------
    t : float
        Step size
    """
    t = 1

    # Precompute some constants
    fx = f(x)
    grad_fx_delta_x = np.dot(grad_f(x), delta_x)

    # Check delta_x is a descent direction
    if grad_fx_delta_x >= 0:
        raise RuntimeError("delta_x isn't a descent direction!")

    # Backtracking loop
    for it in range(max_it):
        if f(x + t * delta_x) <= fx + alpha * t * grad_fx_delta_x:
            break
        t *= beta
    else:
        print("Warning: maximum line search iteration limit reached")

    return t


def newton(x_0, f, grad_f, hess_f, eps=1e-3, alpha=0.1, beta=0.5, max_it=50):
    """
    Newton's method for unconstrained minimisation with backtracking line search.

    Parameters
    ----------
    x_0 : array_like, shape (n,)
        Initial point for the iteration.
    f : callable
        Objective function; f(x) → float.
    grad_f : callable
        Gradient function; grad_f(x) → ndarray of shape (n,).
    hess_f : callable
        Hessian function; hess_f(x) → ndarray of shape (n, n).
    eps : float, optional
        Stopping tolerance based on the Newton decrement λ(x). The solver stops when λ(x)^2 / 2 ≤ eps.
    alpha : float, optional
        Backtracking line search sufficient decrease parameter (0 < alpha < 0.5).
    beta : float, optional
        Backtracking line search shrinkage factor (0 < beta < 1).
    max_it : int, optional
        Maximum number of Newton iterations.

    Returns
    -------
    x : ndarray, shape (n,)
        The estimated minimiser of f.
    it : int
        Number of Newton iterations performed.
    xs : list of ndarray
        Full iterate history including x0 and all accepted updates.
    """
    x = x_0.copy()
    xs = [x.copy()]  # Store initial point

    for it in range(0, max_it):

        # Precompute gradient and Hessian
        g = grad_f(x)
        H = hess_f(x)

        # Newton step
        delta_x = - np.linalg.solve(H, g)

        # Newton decrement
        lmbda_sq = np.dot(delta_x, H @ delta_x)

        if lmbda_sq / 2 <= eps:
            break

        t = line_search(x, delta_x, f, grad_f, alpha, beta)

        x += t * delta_x
        xs.append(x.copy())  # Store updated point
    else:
        print("Warning: maximum Newton iteration limit reached")

    return x, it, xs


def newton_eq(f, grad, hess, A, b, x0, alpha=0.1, beta=0.5, eps=1e-6, max_it=50, feasibility_tol=1e-10):
    """
    Newton's method for equality-constrained minimization using the KKT system.

    Solves: minimize f(x) subject to Ax = b

    At each iteration, solves the KKT system:
    [H(x)   A^T] [Δx]   = -[∇f(x)]
    [A      0  ] [ν ]      [Ax - b]

    where H(x) is the Hessian and ν is the dual variable.

    Parameters
    ----------
    f : callable
        Objective function; f(x) → float.
    grad : callable
        Gradient function; grad(x) → ndarray of shape (n,).
    hess : callable
        Hessian function; hess(x) → ndarray of shape (n, n).
    A : ndarray, shape (p, n)
        Constraint matrix (rank p < n).
    b : ndarray, shape (p,)
        Constraint right-hand side.
    x0 : ndarray, shape (n,)
        Initial feasible point (must satisfy Ax0 = b).
    alpha : float, optional
        Backtracking line search sufficient decrease parameter (0 < alpha < 0.5).
    beta : float, optional
        Backtracking line search shrinkage factor (0 < beta < 1).
    eps : float, optional
        Stopping tolerance. The solver stops when λ²/2 ≤ eps,
        where λ² = Δx^T H(x) Δx is the equality-constrained Newton decrement.
    max_it : int, optional
        Maximum number of Newton iterations.
    feasibility_tol : float, optional
        Numerical tolerance for checking feasibility ||Ax - b||.

    Returns
    -------
    x : ndarray, shape (n,)
        The estimated constrained minimizer of f.
    it : int
        Number of Newton iterations performed.
    xs : list of ndarray
        Full iterate history including x0 and all accepted updates.
    nu_star : ndarray, shape (p,)
        Final dual variable from the last KKT solve.
    """
    x = x0.copy()
    xs = [x.copy()]  # Store initial point
    nu_star = None  # Will store the final dual variable

    # Check initial feasibility
    residual = np.linalg.norm(A @ x - b)
    if residual > feasibility_tol:
        raise ValueError(f"Initial point x0 is not feasible: ||Ax0 - b|| = {residual:.2e} > {feasibility_tol}")

    n = len(x)
    p = A.shape[0]

    for it in range(max_it):
        # Compute gradient and Hessian at current point
        g = grad(x)
        H = hess(x)

        # Build KKT system:
        # [H   A^T] [Δx]   = -[g    ]
        # [A   0  ] [ν ]      [Ax - b]

        # Assemble KKT matrix
        KKT = np.zeros((n + p, n + p))
        KKT[:n, :n] = H
        KKT[:n, n:] = A.T
        KKT[n:, :n] = A
        # KKT[n:, n:] is already zero

        # Assemble right-hand side
        rhs = np.zeros(n + p)
        rhs[:n] = -g
        rhs[n:] = -(A @ x - b)

        # Solve KKT system
        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            print("Warning: KKT system is singular")
            break

        delta_x = sol[:n]
        nu = sol[n:]

        # Store the dual variable
        nu_star = nu

        # Compute equality-constrained Newton decrement
        lmbda_sq = delta_x.T @ H @ delta_x

        # Check stopping criterion
        if lmbda_sq / 2 <= eps:
            break

        # Backtracking line search maintaining feasibility
        # Note: A(x + t*Δx) = Ax + t*A*Δx = b + t*A*Δx
        # From KKT system: A*Δx = -(Ax - b), so A(x + t*Δx) = b + t*(-(Ax-b)) = b(1-t) + t*b = b
        # Thus feasibility is maintained automatically for any t

        t = 1.0
        fx = f(x)
        grad_fx = g  # Already computed
        descent_direction_product = grad_fx.T @ delta_x

        # Backtracking loop
        max_ls_it = 25
        for ls_it in range(max_ls_it):
            x_new = x + t * delta_x

            # Check feasibility (should be maintained, but verify numerically)
            feasibility_error = np.linalg.norm(A @ x_new - b)
            if feasibility_error > feasibility_tol:
                # Project back to feasible set if there's numerical drift
                # Using: x_new = x_new - A^T(AA^T)^{-1}(Ax_new - b)
                AAT_inv = np.linalg.inv(A @ A.T)
                x_new = x_new - A.T @ AAT_inv @ (A @ x_new - b)

            # Check sufficient decrease condition
            if f(x_new) <= fx + alpha * t * descent_direction_product:
                break
            t *= beta
        else:
            print("Warning: maximum line search iteration limit reached")

        # Accept step
        x = x_new
        xs.append(x.copy())
    else:
        print("Warning: maximum Newton iteration limit reached")

    return x, it, xs, nu_star
