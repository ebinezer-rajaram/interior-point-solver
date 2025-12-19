"""
Q1(e): Barrier Method with Phase I for Inequality-Constrained Optimization

This script implements the barrier method for solving:
    minimize f0(x1, x2) = log(e^x1 + e^x2) + 0.5*(x1^2 + x2^2)
    subject to:
        0.5*x1 + x2 = 1                    (equality constraint)
        x1 + x2 - 3/2 <= 0                 (inequality constraint f1)
        -0.25*x1 + x2 - 2 <= 0             (inequality constraint f2)
        -0.1*x1 - x2 + 1 <= 0              (inequality constraint f3)

The barrier method uses the log barrier:
    φ_t(x) = t*f0(x) - Σ log(-fi(x))

Phase I finds an initial strictly feasible point by solving:
    minimize s
    subject to: fi(x) <= s, Ax = b

The implementation includes:
- Barrier method outer loop with parameter t
- Inner loop using equality-constrained Newton's method
- Phase I to find initial feasible point
- Convergence tracking and visualization
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.optimizer import newton_eq
from src.functions import f0, grad_f0, hess_f0


# ============================================================================
# Inequality constraint functions for Q1(e)
# ============================================================================

def f1(x):
    """f1(x) = x1 + x2 - 3/2"""
    return x[0] + x[1] - 1.5


def grad_f1(x):
    """Gradient of f1"""
    return np.array([1.0, 1.0])


def hess_f1(x):
    """Hessian of f1 (zero for affine function)"""
    return np.zeros((2, 2))


def f2(x):
    """f2(x) = -0.25*x1 + x2 - 2"""
    return -0.25 * x[0] + x[1] - 2.0


def grad_f2(x):
    """Gradient of f2"""
    return np.array([-0.25, 1.0])


def hess_f2(x):
    """Hessian of f2 (zero for affine function)"""
    return np.zeros((2, 2))


def f3(x):
    """f3(x) = -0.1*x1 - x2 + 1"""
    return -0.1 * x[0] - x[1] + 1.0


def grad_f3(x):
    """Gradient of f3"""
    return np.array([-0.1, -1.0])


def hess_f3(x):
    """Hessian of f3 (zero for affine function)"""
    return np.zeros((2, 2))


# ============================================================================
# Barrier Method Implementation
# ============================================================================

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
        Equality constraint matrix.
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
        History of outer iterations containing:
            - t: barrier parameter
            - x: current point
            - f0_val: objective value
            - gap: duality gap (m/t)
            - num_inner_it: number of inner Newton iterations
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
        except Exception as e:
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


# ============================================================================
# Phase I Implementation
# ============================================================================

def phase_I(fis, grads, hesses, A, b, n_original, t0=1.0, mu=10.0,
            eps_outer=1e-6, alpha=0.1, beta=0.5, eps_inner=1e-6):
    """
    Phase I method to find strictly feasible point.

    Solves: minimize s
           subject to: fi(x) <= s, i=1..m
                      Ax = b

    Uses variable z = [x; s] where x has dimension n_original and s is scalar.

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

    Returns
    -------
    x_phase1 : ndarray, shape (n_original,) or None
        Strictly feasible point if found.
    s_star : float
        Optimal value of s (if < 0, then x_phase1 is strictly feasible).
    z_star : ndarray, shape (n_original + 1,)
        Full Phase I solution [x; s].
    """
    m = len(fis)

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

    print(f"Phase I initialization:")
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

    print(f"\nPhase I completed:")
    print(f"  Number of outer iterations: {len(history_phase1)}")
    print(f"  s_star = {s_star:.10f}")

    if s_star < 0:
        print(f"  SUCCESS: Found strictly feasible point (s_star < 0)")
        return x_phase1, s_star, z_star
    else:
        print(f"  FAILED: No strictly feasible point found (s_star >= 0)")
        return None, s_star, z_star


# ============================================================================
# Visualization
# ============================================================================

def plot_feasible_region_with_solution(f0, A, b, fis, x_star, history,
                                       x_range=(-1, 2.5), y_range=(-0.5, 2.5),
                                       output_file='plots/q1e_solution.png'):
    """
    Plot the feasible region and barrier method trajectory.

    Parameters
    ----------
    f0 : callable
        Objective function.
    A : ndarray
        Equality constraint matrix.
    b : ndarray
        Equality constraint RHS.
    fis : list of callable
        Inequality constraint functions.
    x_star : ndarray
        Final solution.
    history : list of dict
        Barrier method history.
    x_range, y_range : tuple
        Plot ranges.
    output_file : str
        Output file path.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create grid for contours
    x1_vals = np.linspace(x_range[0], x_range[1], 500)
    x2_vals = np.linspace(y_range[0], y_range[1], 500)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Evaluate objective on grid
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_point = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = f0(x_point)

    # Create figure with professional styling
    fig, ax = plt.subplots(figsize=(12, 9))

    # Use filled contours matching the existing style
    contourf = ax.contourf(X1, X2, Z, levels=40, cmap='RdYlBu_r', alpha=0.8)
    contours = ax.contour(X1, X2, Z, levels=40, colors='black',
                          alpha=0.2, linewidths=0.5)

    # Add colorbar with better formatting
    cbar = plt.colorbar(contourf, ax=ax, label=r'$f_0(x_1, x_2)$', pad=0.02)
    cbar.ax.yaxis.set_label_coords(3.5, 0.5)

    # Compute feasible segment on equality line
    # Equality: x2 = 1 - 0.5*x1
    # For a range of x1, compute x2 and check if all inequalities are satisfied
    x1_samples = np.linspace(x_range[0], x_range[1], 1000)
    x2_samples = b[0] - A[0, 0] * x1_samples

    feasible_x1 = []
    feasible_x2 = []
    for x1_val, x2_val in zip(x1_samples, x2_samples):
        x_point = np.array([x1_val, x2_val])
        if all(fi(x_point) <= 0 for fi in fis):
            feasible_x1.append(x1_val)
            feasible_x2.append(x2_val)

    # Plot inequality constraint boundaries (muted, no legend)
    x1_line = np.linspace(x_range[0], x_range[1], 500)
    x2_line = b[0] - A[0, 0] * x1_line
    x2_f1 = 1.5 - x1_line
    x2_f2 = 2.0 + 0.25 * x1_line
    x2_f3 = 1.0 - 0.1 * x1_line

    ax.plot(x1_line, x2_f1, 'r--', linewidth=2, alpha=0.6, zorder=7)
    ax.plot(x1_line, x2_f2, 'g--', linewidth=2, alpha=0.6, zorder=7)
    ax.plot(x1_line, x2_f3, 'orange', linestyle='--', linewidth=2, alpha=0.6, zorder=7)

    # Plot equality constraint line
    ax.plot(x1_line, x2_line,
            color='gray', linewidth=1.5, linestyle='--',
            alpha=0.4, zorder=8)

    # Highlight feasible segment with thick line
    if len(feasible_x1) > 0:
        ax.plot(feasible_x1, feasible_x2,
                color='#9B59B6', linewidth=4, alpha=0.8,
                zorder=9, label='Feasible segment', solid_capstyle='round')

    # Plot barrier method trajectory
    if len(history) > 0:
        x_hist = np.array([h['x'] for h in history])

        # Draw path with white outline
        ax.plot(x_hist[:, 0], x_hist[:, 1],
                color='white', linewidth=3.5, alpha=0.8, zorder=10)
        ax.plot(x_hist[:, 0], x_hist[:, 1],
                color='#FF6B35', linewidth=2.5, linestyle='-',
                alpha=1.0, zorder=11, label='Barrier path')

        # Mark iterate points
        ax.scatter(x_hist[:, 0], x_hist[:, 1],
                  s=80, c='white', edgecolors='#FF6B35',
                  linewidths=2, zorder=12, alpha=0.9)

        # Add iteration numbers to points
        for i, (x1, x2) in enumerate(x_hist):
            ax.annotate(f'{i}', xy=(x1, x2), xytext=(5, 5),
                       textcoords='offset points', fontsize=9,
                       color='#2C3E50', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='none', alpha=0.7), zorder=13)

        # Mark starting point
        ax.scatter(x_hist[0, 0], x_hist[0, 1], s=200, marker='o',
                  c='#2ECC71', edgecolors='white', linewidths=2.5,
                  label='Start', zorder=14)

    # Mark optimal point
    ax.scatter(x_star[0], x_star[1], s=300, marker='*',
              c='#E74C3C', edgecolors='white', linewidths=2,
              label='Optimal', zorder=15)

    # Axis labels with LaTeX formatting
    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title('Barrier Method Convergence',
                fontsize=14, fontweight='bold', pad=15)

    # Enhanced legend
    ax.legend(loc='upper right', framealpha=0.95, shadow=True,
             fancybox=True, edgecolor='gray', borderpad=1, fontsize=11)

    # Set aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Set limits
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # Tight layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Run Phase I and barrier method on Q1(e) problem."""

    print("=" * 80)
    print("Q1(e): Barrier Method with Phase I")
    print("=" * 80)

    # Problem setup
    A = np.array([[0.5, 1.0]])
    b = np.array([1.0])
    n = 2  # Dimension of x

    # Inequality constraints
    fis = [f1, f2, f3]
    grads = [grad_f1, grad_f2, grad_f3]
    hesses = [hess_f1, hess_f2, hess_f3]
    m = len(fis)

    print(f"\nProblem:")
    print(f"  Objective: f0(x) = log(e^x1 + e^x2) + 0.5*(x1² + x2²)")
    print(f"  Equality:  0.5*x1 + x2 = 1")
    print(f"  Inequality f1: x1 + x2 - 1.5 <= 0")
    print(f"  Inequality f2: -0.25*x1 + x2 - 2 <= 0")
    print(f"  Inequality f3: -0.1*x1 - x2 + 1 <= 0")
    print("=" * 80)

    # ========================================================================
    # Phase I: Find strictly feasible starting point
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE I: Finding Strictly Feasible Point")
    print("=" * 80)

    x_phase1, s_star, z_star = phase_I(
        fis=fis,
        grads=grads,
        hesses=hesses,
        A=A,
        b=b,
        n_original=n,
        t0=1.0,
        mu=10.0,
        eps_outer=1e-6,
        alpha=0.1,
        beta=0.5,
        eps_inner=1e-6
    )

    if x_phase1 is None:
        print("\nPhase I failed: No strictly feasible point exists.")
        return

    # Verify Phase I solution
    print("\n" + "-" * 80)
    print("Phase I Solution:")
    print("-" * 80)
    print(f"x_phase1 = {x_phase1}")
    print(f"s_star   = {s_star:.10f}")
    print(f"\nConstraint verification:")
    print(f"  Equality: A @ x_phase1 = {(A @ x_phase1)[0]:.10f} (target: {b[0]:.1f})")
    print(f"  Inequality f1(x_phase1) = {f1(x_phase1):.10f} (must be < 0)")
    print(f"  Inequality f2(x_phase1) = {f2(x_phase1):.10f} (must be < 0)")
    print(f"  Inequality f3(x_phase1) = {f3(x_phase1):.10f} (must be < 0)")
    print(f"  max_i fi(x_phase1) = {max(f1(x_phase1), f2(x_phase1), f3(x_phase1)):.10f}")

    # Check strict feasibility
    all_negative = all(fi(x_phase1) < 0 for fi in fis)
    equality_satisfied = np.abs((A @ x_phase1 - b)[0]) < 1e-8

    if all_negative and equality_satisfied:
        print(f"\nPhase I successful: x_phase1 is strictly feasible!")
    else:
        print(f"\nPhase I verification failed!")
        return

    # ========================================================================
    # Main Barrier Method: Solve original problem
    # ========================================================================
    print("\n" + "=" * 80)
    print("MAIN BARRIER METHOD: Solving Original Problem")
    print("=" * 80)

    x_star, history = barrier_method(
        f0=f0,
        grad0=grad_f0,
        hess0=hess_f0,
        fis=fis,
        grads=grads,
        hesses=hesses,
        A=A,
        b=b,
        x0=x_phase1,
        t0=1.0,
        mu=10.0,
        eps_outer=1e-6,
        alpha=0.1,
        beta=0.5,
        eps_inner=1e-6
    )

    # Print convergence history
    print("\n" + "-" * 80)
    print("Barrier Method Convergence History:")
    print("-" * 80)
    print(f"{'Iter':>4} {'t':>12} {'f0(x)':>12} {'Gap (m/t)':>12} {'Inner Its':>10}")
    print("-" * 80)
    for i, h in enumerate(history):
        print(f"{i:4d} {h['t']:12.4e} {h['f0_val']:12.6f} {h['gap']:12.4e} {h['num_inner_it']:10d}")
    print("-" * 80)

    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"x_star = {x_star}")
    print(f"f0(x_star) = {f0(x_star):.10f}")
    print(f"\nConstraint verification:")
    print(f"  Equality: A @ x_star = {(A @ x_star)[0]:.10f} (target: {b[0]:.1f})")
    print(f"  Inequality f1(x_star) = {f1(x_star):.10f} (must be <= 0)")
    print(f"  Inequality f2(x_star) = {f2(x_star):.10f} (must be <= 0)")
    print(f"  Inequality f3(x_star) = {f3(x_star):.10f} (must be <= 0)")
    print(f"\nNumber of outer iterations: {len(history)}")
    print(f"Final duality gap: {history[-1]['gap']:.4e}")
    print("=" * 80)

    # Create visualization - zoom out slightly to show context
    plot_feasible_region_with_solution(
        f0=f0,
        A=A,
        b=b,
        fis=fis,
        x_star=x_star,
        history=history,
        x_range=(-1, 2.5),
        y_range=(-0.5, 2.5),
        output_file='plots/q1e_solution.png'
    )

    # ========================================================================
    # Compare with SciPy Reference Optimizer
    # ========================================================================
    print("\n" + "=" * 80)
    print("SciPy Reference Optimizer (SLSQP)")
    print("=" * 80)

    from scipy.optimize import minimize

    # Define constraints in SciPy format
    def constraint_eq(x):
        return (A @ x - b)[0]

    def constraint_ineq_combined(x):
        # Return array of -f_i(x) so that -f_i(x) >= 0 means f_i(x) <= 0
        return np.array([-f1(x), -f2(x), -f3(x)])

    constraints = [
        {'type': 'eq', 'fun': constraint_eq},
        {'type': 'ineq', 'fun': constraint_ineq_combined}
    ]

    # Run SciPy optimizer
    scipy_result = minimize(
        fun=f0,
        x0=x_phase1,
        method='SLSQP',
        jac=grad_f0,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )

    x_scipy = scipy_result.x
    f_scipy = scipy_result.fun

    # Compute differences
    solution_diff = np.linalg.norm(x_star - x_scipy)
    objective_diff = abs(f0(x_star) - f_scipy)

    print(f"SciPy solution:              {x_scipy}")
    print(f"SciPy f0(x):                 {f_scipy:.10f}")
    print(f"SciPy A @ x:                 {(A @ x_scipy)[0]:.10f}")
    print(f"SciPy f1(x):                 {f1(x_scipy):.10f}")
    print(f"SciPy f2(x):                 {f2(x_scipy):.10f}")
    print(f"SciPy f3(x):                 {f3(x_scipy):.10f}")
    print(f"SciPy iterations:            {scipy_result.nit}")
    print(f"\nComparison:")
    print(f"||x_barrier - x_scipy||:     {solution_diff:.4e}")
    print(f"|f_barrier - f_scipy|:       {objective_diff:.4e}")
    print("=" * 80)


if __name__ == "__main__":
    main()
