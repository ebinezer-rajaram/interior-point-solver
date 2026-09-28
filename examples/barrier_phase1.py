"""
Barrier Method with Phase I for Inequality-Constrained Optimization

This example applies the barrier method in src/barrier.py to:
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

The example:
- runs Phase I to find a strictly feasible starting point
- runs the barrier method from that point (outer loop over t, inner
  equality-constrained Newton centering steps)
- prints the convergence history and cross-checks against SciPy (SLSQP)
- plots the feasible segment and barrier path to plots/barrier_solution.png
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from src.barrier import barrier_method, phase_I
from src.functions import f0, grad_f0, hess_f0


# ============================================================================
# Inequality constraint functions
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
# Visualization
# ============================================================================

def plot_feasible_region_with_solution(f0, A, b, fis, x_star, history,
                                       x_range=(-1, 2.5), y_range=(-0.5, 2.5),
                                       output_file='plots/barrier_solution.png'):
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
    """Run Phase I and the barrier method on the example problem."""

    print("=" * 80)
    print("Barrier Method with Phase I")
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
        eps_inner=1e-6,
        verbose=True
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
        output_file='plots/barrier_solution.png'
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
