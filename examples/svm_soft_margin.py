"""
Soft-Margin SVM using the Barrier Method

Solve the soft-margin SVM problem:
    min_{w,b,ξ} 1/2 ||w||_2^2 + c * ||ξ||_1
    s.t. y_i(w^T x_i + b) >= 1 - ξ_i, i=1,...,m
         ξ_i >= 0, i=1,...,m

Reformulated for barrier method (fi(x) <= 0):
    min_{w,b,ξ} 1/2 ||w||_2^2 + c * Σξ_i
    s.t. 1 - ξ_i - y_i(w^T x_i + b) <= 0, i=1,...,m
         -ξ_i <= 0, i=1,...,m

Decision variables: x = [w1, w2, b, ξ_1, ..., ξ_m]^T
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.barrier import barrier_method


def generate_data(n_samples=100, random_state=42):
    """
    Generate a 2D dataset where misclassifications decrease as c increases.

    Strategy: Create two well-separated clusters, then add ambiguous points
    near the boundary that high c values will correctly classify by shifting
    the decision boundary.

    Returns:
        X: array of shape (n_samples, 2) - feature vectors
        y: array of shape (n_samples,) - labels in {-1, +1}
    """
    rng = np.random.default_rng(random_state)

    # Create two well-separated main clusters
    centers = [[-2.0, -2.0], [2.0, 2.0]]
    X_main, y_binary = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=0.7,  # Tighter clusters
        random_state=random_state
    )
    y_main = 2 * y_binary - 1

    # Add strategically placed ambiguous points:
    # These are labeled correctly but positioned such that:
    # - Low c: boundary prioritizes large margin, some misclassified
    # - High c: boundary shifts to classify them correctly

    # Add positive class (+1) points slightly on the negative side of the natural boundary
    # (these will be misclassified at low c but correctly classified at high c)
    n_ambiguous = 8
    ambiguous_pos = rng.uniform(low=[-1.5, -1.5], high=[0.2, 0.2], size=(n_ambiguous, 2))

    # Add negative class (-1) points slightly on the positive side
    ambiguous_neg = rng.uniform(low=[-0.2, -0.2], high=[1.5, 1.5], size=(n_ambiguous, 2))

    # Combine all data
    X = np.vstack([X_main, ambiguous_pos, ambiguous_neg])
    y = np.concatenate([y_main, np.ones(n_ambiguous), -np.ones(n_ambiguous)])

    return X, y


def soft_margin_svm_barrier(X, y, c, t0=1.0, mu=10.0, eps_outer=1e-6,
                            alpha=0.1, beta=0.5, eps_inner=1e-6, max_outer_it=100,
                            verbose=False):
    """
    Solve soft-margin SVM using barrier method.

    Variables: z = [w1, w2, b, ξ_1, ..., ξ_m]^T

    Objective: f0(z) = 1/2 * (w1^2 + w2^2) + c * Σξ_i

    Constraints:
        fi(z) = 1 - ξ_i - y_i(w^T x_i + b) <= 0,  i=1,...,m  (margin constraints)
        gi(z) = -ξ_i <= 0,                        i=1,...,m  (non-negativity)

    Parameters:
        X: array of shape (m, 2) - training features
        y: array of shape (m,) - training labels in {-1, +1}
        c: float - soft-margin penalty parameter
        t0: float - initial barrier parameter
        mu: float - barrier parameter increase factor
        eps_outer: float - outer loop convergence tolerance
        alpha: float - backtracking line search parameter
        beta: float - backtracking shrinkage factor
        eps_inner: float - Newton method convergence tolerance
        max_outer_it: int - maximum outer iterations
        verbose: bool - whether to print iteration details

    Returns:
        z_star: optimal solution [w1, w2, b, ξ_1, ..., ξ_m]
        n_iter: number of outer iterations
        history: list of iterates at each outer iteration
    """
    m = len(y)
    n = 3 + m  # dimension: [w1, w2, b, ξ_1, ..., ξ_m]

    # Objective: f0(z) = 1/2 * ||w||^2 + c * Σξ_i
    def f0(z):
        w = z[:2]
        xi = z[3:]
        return 0.5 * np.dot(w, w) + c * np.sum(xi)

    def grad0(z):
        g = np.zeros(n)
        g[:2] = z[:2]  # gradient w.r.t. w
        # gradient w.r.t. b is 0
        g[3:] = c  # gradient w.r.t. ξ
        return g

    def hess0(z):
        H = np.zeros((n, n))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        # All other entries are 0 (linear in ξ, constant in b)
        return H

    # Create constraint functions
    fis = []
    grads = []
    hesses = []

    # Margin constraints: fi(z) = 1 - ξ_i - y_i(w^T x_i + b) <= 0
    for i in range(m):
        xi_data = X[i]
        yi = y[i]

        def make_margin_constraint(i=i, xi_data=xi_data, yi=yi):
            def fi(z):
                w = z[:2]
                b = z[2]
                xi_slack = z[3 + i]
                return 1.0 - xi_slack - yi * (np.dot(w, xi_data) + b)

            def grad_fi(z):
                g = np.zeros(n)
                g[:2] = -yi * xi_data  # gradient w.r.t. w
                g[2] = -yi              # gradient w.r.t. b
                g[3 + i] = -1.0         # gradient w.r.t. ξ_i
                return g

            def hess_fi(z):
                # Second derivative is zero (linear constraint)
                return np.zeros((n, n))

            return fi, grad_fi, hess_fi

        fi, grad_fi, hess_fi = make_margin_constraint()
        fis.append(fi)
        grads.append(grad_fi)
        hesses.append(hess_fi)

    # Non-negativity constraints: gi(z) = -ξ_i <= 0
    for i in range(m):
        def make_nonneg_constraint(i=i):
            def gi(z):
                return -z[3 + i]

            def grad_gi(z):
                g = np.zeros(n)
                g[3 + i] = -1.0
                return g

            def hess_gi(z):
                return np.zeros((n, n))

            return gi, grad_gi, hess_gi

        gi, grad_gi, hess_gi = make_nonneg_constraint()
        fis.append(gi)
        grads.append(grad_gi)
        hesses.append(hess_gi)

    # No equality constraints
    A = np.zeros((0, n))
    b_eq = np.zeros(0)

    # Find strictly feasible starting point
    z0 = find_feasible_start_soft(X, y)

    if verbose:
        print(f"Starting point: w = [{z0[0]:.4f}, {z0[1]:.4f}], b = {z0[2]:.4f}")
        print(f"  Slack variables: min(ξ) = {np.min(z0[3:]):.4f}, max(ξ) = {np.max(z0[3:]):.4f}")
        print(f"Initial objective: f0(z0) = {f0(z0):.6f}")

    # Verify strict feasibility
    max_constraint = max(fi(z0) for fi in fis)
    if verbose:
        print(f"Max constraint value: {max_constraint:.6e} (must be < 0)")
    assert max_constraint < 0, f"Starting point is not strictly feasible! max_constraint = {max_constraint}"

    # Barrier method (shared implementation in src/barrier.py)
    z, outer_history = barrier_method(
        f0, grad0, hess0, fis, grads, hesses, A, b_eq, z0,
        t0=t0, mu=mu, eps_outer=eps_outer, alpha=alpha, beta=beta,
        eps_inner=eps_inner, max_outer_it=max_outer_it
    )
    history = [z0.copy()] + [h['x'] for h in outer_history]

    if verbose:
        print("\n" + "="*80)
        print(f"{'Iter':<6} {'t':<12} {'(2m)/t':<12} {'f0(z)':<12} {'||w||':<12} {'Σξ':<12}")
        print("="*80)
        for k, h in enumerate(outer_history):
            w_norm = np.linalg.norm(h['x'][:2])
            xi_sum = np.sum(h['x'][3:])
            print(f"{k+1:<6} {h['t']:<12.2e} {h['gap']:<12.2e} {h['f0_val']:<12.6f} {w_norm:<12.6f} {xi_sum:<12.6f}")
        print("="*80)
        if outer_history and outer_history[-1]['gap'] <= eps_outer:
            print(f"Converged: (2m)/t = {outer_history[-1]['gap']:.2e} <= {eps_outer:.2e}")
        else:
            print("Warning: barrier method did not reach (2m)/t <= eps_outer")

    return z, len(outer_history), history


def find_feasible_start_soft(X, y):
    """
    Find a strictly feasible starting point for the soft-margin SVM problem.

    Strategy:
    1. Find a simple separating hyperplane (using class centroids)
    2. Compute required slack for each point
    3. Add safety margin to ensure strict feasibility

    Returns:
        z0: initial point [w1, w2, b, ξ_1, ..., ξ_m]
    """
    m = len(y)

    # Use class centroids to find initial separator
    pos_idx = y == 1
    neg_idx = y == -1

    centroid_pos = np.mean(X[pos_idx], axis=0)
    centroid_neg = np.mean(X[neg_idx], axis=0)

    # Normal vector points from negative to positive centroid
    w_init = centroid_pos - centroid_neg
    w_init = w_init / (np.linalg.norm(w_init) + 1e-10)

    # Find b such that the hyperplane passes between centroids
    b_init = -0.5 * np.dot(w_init, centroid_pos + centroid_neg)

    # Compute margins for all points
    margins = y * (X @ w_init + b_init)

    # Compute required slack variables: ξ_i = max(0, 1 - margin_i)
    xi_init = np.maximum(0, 1.0 - margins)

    # Add safety margin to ensure strict feasibility
    xi_init += 0.5

    # Construct initial point
    z0 = np.zeros(3 + m)
    z0[:2] = w_init
    z0[2] = b_init
    z0[3:] = xi_init

    # Verify strict feasibility of both constraint types
    # Margin constraints: 1 - ξ_i - y_i(w^T x_i + b) < 0
    margin_violations = 1.0 - xi_init - margins
    assert np.all(margin_violations < 0), f"Margin constraints not satisfied: max = {np.max(margin_violations)}"

    # Non-negativity constraints: -ξ_i < 0
    assert np.all(xi_init > 0), f"Non-negativity constraints not satisfied: min(ξ) = {np.min(xi_init)}"

    return z0


def compute_metrics(X, y, w, b, xi):
    """
    Compute various metrics for the soft-margin SVM solution.

    Returns:
        metrics: dict with keys:
            - w_norm: ||w||_2
            - margin: 1/||w||_2
            - n_misclassified: number of misclassified points
            - n_margin_violations: number of points with y_i(w^T x_i + b) < 1
            - xi_sum: Σξ_i
            - xi_max: max(ξ_i)
    """
    w_norm = np.linalg.norm(w)
    margin = 1.0 / w_norm if w_norm > 0 else np.inf

    # Classification: sign(w^T x + b)
    predictions = np.sign(X @ w + b)
    n_misclassified = np.sum(predictions != y)

    # Margin violations: y_i(w^T x_i + b) < 1
    functional_margins = y * (X @ w + b)
    n_margin_violations = np.sum(functional_margins < 1.0)

    # Slack statistics
    xi_sum = np.sum(xi)
    xi_max = np.max(xi)

    return {
        'w_norm': w_norm,
        'margin': margin,
        'n_misclassified': n_misclassified,
        'n_margin_violations': n_margin_violations,
        'xi_sum': xi_sum,
        'xi_max': xi_max
    }


def plot_decision_boundary(X, y, w, b, save_path):
    """
    Plot dataset with decision boundary and margins.
    """
    # Set professional plot styling
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['lines.linewidth'] = 1.5

    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute plot range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_vals = np.linspace(x_min, x_max, 500)

    # Decision boundary: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b)/w2
    y_decision = -(w[0] * x_vals + b) / w[1]

    # Margin boundaries: w^T x + b = ±1
    y_margin_pos = -(w[0] * x_vals + b - 1) / w[1]
    y_margin_neg = -(w[0] * x_vals + b + 1) / w[1]

    # Shade the margin region
    ax.fill_between(x_vals, y_margin_neg, y_margin_pos,
                    alpha=0.08, color='purple', zorder=1)

    # Plot margin lines
    ax.plot(x_vals, y_margin_pos, color='#9B59B6', linewidth=2,
            linestyle='--', alpha=0.8, zorder=8,
            label=r'$\mathbf{w}^\top\mathbf{x} + b = \pm 1$')
    ax.plot(x_vals, y_margin_neg, color='#9B59B6', linewidth=2,
            linestyle='--', alpha=0.8, zorder=8)

    # Plot decision boundary
    ax.plot(x_vals, y_decision, color='black', linewidth=2.5,
            linestyle='-', alpha=0.9, zorder=9,
            label=r'$\mathbf{w}^\top\mathbf{x} + b = 0$')

    # Plot training points
    pos_idx = y == 1
    neg_idx = y == -1

    ax.scatter(X[pos_idx, 0], X[pos_idx, 1], c='#E74C3C', marker='o', s=100,
               label='Class $+1$', edgecolors='darkred', linewidth=1.2,
               zorder=10, alpha=0.85)
    ax.scatter(X[neg_idx, 0], X[neg_idx, 1], c='#3498DB', marker='s', s=100,
               label='Class $-1$', edgecolors='darkblue', linewidth=1.2,
               zorder=10, alpha=0.85)

    # Axis labels
    ax.set_xlabel(r'$x_1$', fontsize=11)
    ax.set_ylabel(r'$x_2$', fontsize=11)
    ax.set_title('Soft-Margin SVM ($c = 1.0$)', fontsize=12, fontweight='bold', pad=10)

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Legend - fully opaque background to avoid overlapping with points, on top
    legend = ax.legend(loc='upper right', framealpha=1, edgecolor='black', fancybox=False, facecolor='white')
    legend.set_zorder(100)

    # Grid
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')

    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Tight layout
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {save_path}")
    plt.close(fig)


def plot_metrics_vs_c(c_values, results, save_path):
    """
    Plot how margin and misclassifications change as a function of c.
    """
    # Set professional plot styling
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Extract metrics
    margins = [r['margin'] for r in results]
    n_misclassified = [r['n_misclassified'] for r in results]
    n_violations = [r['n_margin_violations'] for r in results]
    xi_sums = [r['xi_sum'] for r in results]

    # Plot 1: Margin vs c
    ax = axes[0, 0]
    ax.semilogx(c_values, margins, 'o-', color='#2C3E50', linewidth=2, markersize=8)
    ax.set_xlabel(r'$c$ (penalty parameter)', fontsize=11)
    ax.set_ylabel(r'Margin $r = 1/\|\mathbf{w}\|_2$', fontsize=11)
    ax.set_title('Margin vs. Penalty Parameter', fontsize=12, fontweight='normal')
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')

    # Plot 2: Misclassifications vs c
    ax = axes[0, 1]
    ax.semilogx(c_values, n_misclassified, 's-', color='#E74C3C', linewidth=2, markersize=8)
    ax.set_xlabel(r'$c$ (penalty parameter)', fontsize=11)
    ax.set_ylabel('Number of Misclassifications', fontsize=11)
    ax.set_title('Misclassifications vs. Penalty Parameter', fontsize=12, fontweight='normal')
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')

    # Plot 3: Margin violations vs c
    ax = axes[1, 0]
    ax.semilogx(c_values, n_violations, '^-', color='#9B59B6', linewidth=2, markersize=8)
    ax.set_xlabel(r'$c$ (penalty parameter)', fontsize=11)
    ax.set_ylabel('Number of Margin Violations', fontsize=11)
    ax.set_title('Margin Violations vs. Penalty Parameter', fontsize=12, fontweight='normal')
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')

    # Plot 4: Total slack vs c
    ax = axes[1, 1]
    ax.loglog(c_values, xi_sums, 'd-', color='#16A085', linewidth=2, markersize=8)
    ax.set_xlabel(r'$c$ (penalty parameter)', fontsize=11)
    ax.set_ylabel(r'Total Slack $\sum_i \xi_i$', fontsize=11)
    ax.set_title('Total Slack vs. Penalty Parameter', fontsize=12, fontweight='normal')
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.set_facecolor('white')

    # White background
    fig.patch.set_facecolor('white')

    # Tight layout
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {save_path}")
    plt.close(fig)


def main():
    print("="*80)
    print("Soft-Margin SVM")
    print("="*80)

    # Generate dataset (deterministic)
    np.random.seed(42)
    X, y = generate_data(n_samples=100, random_state=42)
    m = len(y)

    print(f"\nGenerated {m} training samples")
    print(f"  Positive class (+1): {np.sum(y == 1)} samples")
    print(f"  Negative class (-1): {np.sum(y == -1)} samples")

    # List of c values to experiment with
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    print(f"\nRunning experiments for c values: {c_values}")
    print("="*80)

    results = []
    solutions = []

    for c in c_values:
        print(f"\n{'='*80}")
        print(f"Solving for c = {c}")
        print(f"{'='*80}")

        # Solve soft-margin SVM
        z_star, n_iter, history = soft_margin_svm_barrier(
            X, y, c=c,
            t0=1.0,
            mu=10.0,
            eps_outer=1e-6,
            alpha=0.1,
            beta=0.5,
            eps_inner=1e-6,
            max_outer_it=100,
            verbose=True
        )

        # Extract solution components
        w_star = z_star[:2]
        b_star = z_star[2]
        xi_star = z_star[3:]

        # Compute metrics
        metrics = compute_metrics(X, y, w_star, b_star, xi_star)
        metrics['c'] = c
        metrics['n_iter'] = n_iter
        results.append(metrics)
        solutions.append((w_star, b_star, xi_star))

        # Print results
        print(f"\nResults for c = {c}:")
        print(f"  w = [{w_star[0]:.6f}, {w_star[1]:.6f}]")
        print(f"  b = {b_star:.6f}")
        print(f"  ||w||_2 = {metrics['w_norm']:.6f}")
        print(f"  Margin r = 1/||w||_2 = {metrics['margin']:.6f}")
        print(f"  Misclassified points: {metrics['n_misclassified']}")
        print(f"  Margin violations: {metrics['n_margin_violations']}")
        print(f"  Σξ_i = {metrics['xi_sum']:.6f}")
        print(f"  max(ξ_i) = {metrics['xi_max']:.6f}")
        print(f"  Outer iterations: {n_iter}")

    # Print summary table
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"{'c':<10} {'||w||':<12} {'Margin':<12} {'Misclass':<12} {'Violations':<12} {'Σξ':<12} {'max(ξ)':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['c']:<10.2f} {r['w_norm']:<12.6f} {r['margin']:<12.6f} "
              f"{r['n_misclassified']:<12d} {r['n_margin_violations']:<12d} "
              f"{r['xi_sum']:<12.6f} {r['xi_max']:<12.6f}")
    print("="*80)

    # Generate plots
    print("\n" + "="*80)
    print("Generating Plots")
    print("="*80)

    # Plot 1: Decision boundary for c=1.0
    c_idx = c_values.index(1.0)
    w_plot, b_plot, _ = solutions[c_idx]
    plot_decision_boundary(
        X, y, w_plot, b_plot,
        save_path='plots/svm_soft_margin_boundary.png'
    )

    # Plot 2: Metrics vs c
    plot_metrics_vs_c(
        c_values, results,
        save_path='plots/svm_soft_margin_metrics_vs_c.png'
    )

    print("\n" + "="*80)
    print("Done.")
    print("="*80)


if __name__ == '__main__':
    main()
