"""
Log-barrier interior-point method and Phase I feasibility solver.

Solves problems of the form

    minimize    f0(x)
    subject to  fi(x) <= 0,  i = 1..m
                Ax = b

by repeatedly minimising the barrier objective

    phi_t(x) = t*f0(x) - sum_i log(-fi(x))

subject to Ax = b (the centering step, solved with equality-constrained
Newton), then increasing t geometrically until the duality-gap bound m/t
falls below a tolerance.

A strictly feasible starting point can be obtained with `phase_I`, which
applies the same barrier machinery to

    minimize    s
    subject to  fi(x) <= s,  Ax = b.
"""

import numpy as np

from .optimizer import newton_eq


def barrier_method(f0, grad0, hess0, fis, grads, hesses, A, b, x0,
                   t0=1.0, mu=10.0, eps_outer=1e-6, alpha=0.1, beta=0.5,
                   eps_inner=1e-6, max_outer_it=100):
    """
    Barrier method for inequality-constrained optimization.

    Solves: minimize f0(x)
           subject to: fi(x) <= 0, i=1..m
                      Ax = b

    Uses the log barrier objective:
        φ_t(x) = t*f0(x) - Σ log(-fi(x))

    Parameters
    ----------
    f0 : callable
        Objective function; f0(x) → float.
    grad0 : callable
        Gradient of f0; grad0(x) → ndarray of shape (n,).
    hess0 : callable
        Hessian of f0; hess0(x) → ndarray of shape (n, n).
    fis : list of callable
        List of inequality constraint functions; fi(x) → float.
        Constraints are fi(x) <= 0.
    grads : list of callable
        List of gradient functions for constraints.
    hesses : list of callable
        List of Hessian functions for constraints.
    A : ndarray, shape (p, n)
        Equality constraint matrix. Use shape (0, n) when there are no
        equality constraints.
    b : ndarray, shape (p,)
        Equality constraint right-hand side.
    x0 : ndarray, shape (n,)
        Initial strictly feasible point (fi(x0) < 0 for all i, Ax0 = b).
    t0 : float, optional
        Initial barrier parameter.
    mu : float, optional
        Barrier parameter multiplier (mu > 1).
    eps_outer : float, optional
        Outer loop stopping tolerance (m/t <= eps_outer).
    alpha : float, optional
        Line search sufficient decrease parameter.
    beta : float, optional
        Line search shrinkage factor.
    eps_inner : float, optional
        Inner loop (Newton) stopping tolerance.
    max_outer_it : int, optional
        Maximum number of outer iterations.

    Returns
    -------
    x_star : ndarray, shape (n,)
        Optimal solution.
    history : list of dict
        One entry per completed outer iteration (centering step), containing:
            - t: barrier parameter
            - x: current point
            - f0_val: objective value
            - gap: duality gap (m/t)
            - num_inner_it: number of inner Newton iterations
            - constraint_vals: list of fi(x)
    """
    m = len(fis)
    x = x0.copy()
    t = t0
    history = []

    for outer_it in range(max_outer_it):
        # Check if all constraints are strictly satisfied
        constraint_violations = [fi(x) for fi in fis]
        if any(cv >= 0 for cv in constraint_violations):
            raise ValueError(f"Point not strictly feasible at outer iteration {outer_it}: "
                           f"constraint values = {constraint_violations}")

        # Define barrier objective φ_t(x) = t*f0(x) - Σ log(-fi(x))
        def phi_t(x_val):
            obj = t * f0(x_val)
            for fi in fis:
                fi_val = fi(x_val)
                if fi_val >= 0:
                    return np.inf  # Not in domain
                obj -= np.log(-fi_val)
            return obj

        # Gradient of φ_t:
        # ∇φ_t = t*∇f0 - Σ (∇fi / (-fi))
        def grad_phi_t(x_val):
            grad = t * grad0(x_val)
            for fi, grad_fi in zip(fis, grads):
                fi_val = fi(x_val)
                grad += grad_fi(x_val) / (-fi_val)  # Note: -∇(-log(-fi)) = ∇fi/(-fi)
            return grad

        # Hessian of φ_t:
        # ∇²φ_t = t*∇²f0 + Σ [∇²fi/(-fi) + (∇fi ∇fi^T)/(-fi)²]
        def hess_phi_t(x_val):
            hess = t * hess0(x_val)
            for fi, grad_fi, hess_fi in zip(fis, grads, hesses):
                fi_val = fi(x_val)
                grad_fi_val = grad_fi(x_val)
                hess += hess_fi(x_val) / (-fi_val)
                hess += np.outer(grad_fi_val, grad_fi_val) / (fi_val ** 2)
            return hess

        # Solve centering problem: minimize φ_t(x) subject to Ax = b
        # using equality-constrained Newton
        try:
            x, num_inner_it, _, _ = newton_eq(
                f=phi_t,
                grad=grad_phi_t,
                hess=hess_phi_t,
                A=A,
                b=b,
                x0=x,
                alpha=alpha,
                beta=beta,
                eps=eps_inner,
                max_it=50
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Warning: Newton solver failed at outer iteration {outer_it}: {e}")
            break

        # Compute duality gap
        gap = m / t

        # Store history
        f0_val = f0(x)
        history.append({
            't': t,
            'x': x.copy(),
            'f0_val': f0_val,
            'gap': gap,
            'num_inner_it': num_inner_it,
            'constraint_vals': [fi(x) for fi in fis]
        })

        # Check stopping criterion
        if gap <= eps_outer:
            break

        # Update barrier parameter
        t *= mu
    else:
        print("Warning: maximum outer iteration limit reached")

    return x, history


def phase_I(fis, grads, hesses, A, b, n_original, t0=1.0, mu=10.0,
            eps_outer=1e-6, alpha=0.1, beta=0.5, eps_inner=1e-6,
            verbose=False):
    """
    Phase I method to find a strictly feasible point.

    Solves: minimize s
           subject to: fi(x) <= s, i=1..m
                      Ax = b

    Uses variable z = [x; s] where x has dimension n_original and s is scalar.
    The starting point is the minimum-norm solution of Ax = b with
    s_0 = max_i fi(x) + 1, which is strictly feasible for the Phase I problem.

    Parameters
    ----------
    fis : list of callable
        List of inequality constraint functions for original problem.
    grads : list of callable
        List of gradient functions for constraints.
    hesses : list of callable
        List of Hessian functions for constraints.
    A : ndarray, shape (p, n_original)
        Equality constraint matrix.
    b : ndarray, shape (p,)
        Equality constraint right-hand side.
    n_original : int
        Dimension of x in original problem.
    t0, mu, eps_outer, alpha, beta, eps_inner : optional
        Parameters for barrier method (see barrier_method).
    verbose : bool, optional
        Print the initialisation and outcome.

    Returns
    -------
    x_phase1 : ndarray, shape (n_original,) or None
        Strictly feasible point if found (s_star < 0), otherwise None.
    s_star : float
        Optimal value of s (if < 0, then x_phase1 is strictly feasible).
    z_star : ndarray, shape (n_original + 1,)
        Full Phase I solution [x; s].
    """
    # Step 1: Find minimum-norm feasible point for Ax = b
    if A.size > 0 and b.size > 0:
        AAT = A @ A.T
        AAT_inv = np.linalg.inv(AAT)
        x_init = A.T @ AAT_inv @ b
    else:
        x_init = np.zeros(n_original)

    # Step 2: Set s_0 = max_i fi(x_init) + 1 to ensure gi(z0) < 0
    max_constraint = max(fi(x_init) for fi in fis)
    s_0 = max_constraint + 1.0

    if verbose:
        print("Phase I initialization:")
        print(f"  x_init = {x_init}")
        print(f"  max_i fi(x_init) = {max_constraint:.6f}")
        print(f"  s_0 = {s_0:.6f}")

    # Initial point for Phase I
    z0 = np.concatenate([x_init, [s_0]])

    # Define Phase I objective: f0^(I)(z) = s
    def f0_phase1(z):
        return z[-1]  # Just s

    def grad0_phase1(z):
        g = np.zeros(n_original + 1)
        g[-1] = 1.0  # ∂s/∂s = 1
        return g

    def hess0_phase1(z):
        return np.zeros((n_original + 1, n_original + 1))

    # Define Phase I constraints: gi(z) = fi(x) - s <= 0
    def make_gi(fi):
        def gi(z):
            x = z[:n_original]
            s = z[-1]
            return fi(x) - s
        return gi

    def make_grad_gi(grad_fi):
        def grad_gi(z):
            x = z[:n_original]
            grad = np.zeros(n_original + 1)
            grad[:n_original] = grad_fi(x)
            grad[-1] = -1.0
            return grad
        return grad_gi

    def make_hess_gi(hess_fi):
        def hess_gi(z):
            x = z[:n_original]
            hess = np.zeros((n_original + 1, n_original + 1))
            hess[:n_original, :n_original] = hess_fi(x)
            # ∂²/∂s² = 0, ∂²/∂x∂s = 0
            return hess
        return hess_gi

    # Create augmented constraint lists
    gis = [make_gi(fi) for fi in fis]
    grad_gis = [make_grad_gi(grad_fi) for grad_fi in grads]
    hess_gis = [make_hess_gi(hess_fi) for hess_fi in hesses]

    # Augmented equality constraint: [A 0] * z = b
    if A.size > 0:
        A_phase1 = np.column_stack([A, np.zeros((A.shape[0], 1))])
    else:
        A_phase1 = np.zeros((0, n_original + 1))

    # Run barrier method on Phase I problem
    if verbose:
        print("\nRunning Phase I barrier method...")
    z_star, history_phase1 = barrier_method(
        f0=f0_phase1,
        grad0=grad0_phase1,
        hess0=hess0_phase1,
        fis=gis,
        grads=grad_gis,
        hesses=hess_gis,
        A=A_phase1,
        b=b,
        x0=z0,
        t0=t0,
        mu=mu,
        eps_outer=eps_outer,
        alpha=alpha,
        beta=beta,
        eps_inner=eps_inner
    )

    # Extract solution
    x_phase1 = z_star[:n_original]
    s_star = z_star[-1]

    if verbose:
        print("\nPhase I completed:")
        print(f"  Number of outer iterations: {len(history_phase1)}")
        print(f"  s_star = {s_star:.10f}")

    if s_star < 0:
        if verbose:
            print("  SUCCESS: Found strictly feasible point (s_star < 0)")
        return x_phase1, s_star, z_star
    else:
        if verbose:
            print("  FAILED: No strictly feasible point found (s_star >= 0)")
        return None, s_star, z_star
