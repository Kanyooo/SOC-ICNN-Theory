import torch
import time


# ============================================================
# Experiment 3: Degeneracy diagnostic for SOC-ICNN
# ============================================================
#
# Purpose:
#   Verify the set-valued first-order geometry at a degenerate point.
#
# We construct a tiny SOC-ICNN with:
#   1. one ReLU preactivation exactly zero at x0,
#   2. one conic residual u(x0)=0.
#
# At x0, the subdifferential is set-valued. We verify:
#
#   (A) The canonical selector is a valid subgradient:
#       f(y) >= f(x0) + g_can^T (y - x0).
#
#   (B) Directional derivative is given by:
#       f'(x0; d) = max_{xi in D*(x0)} G(xi)^T d.
#
#   (C) Random dual readouts never exceed the exact max-readout
#       directional derivative.
#
#   (D) The canonical dual selector is the minimum-norm branch
#       among sampled optimal dual branches.
#
# ============================================================


class DegenerateSOCICNN:
    """
    A tiny hand-crafted SOC-ICNN:

        f(x)
        =
        sum_i c_i ReLU(W_i x + b_i)
        + v^T x
        + b0
        + alpha / 2 * ||B x + e||^2
        + lambda * ||A x + d||.

    At x0 = 0:
        - the first ReLU unit is degenerate because b_0 = 0;
        - the second ReLU unit is active because b_1 > 0;
        - the third ReLU unit is inactive because b_2 < 0;
        - the conic residual is degenerate because A x0 + d = 0.

    This gives both structural degeneracy sources in one example.
    """

    def __init__(self, dtype=torch.float64, device="cpu"):
        self.dtype = dtype
        self.device = device

        self.d0 = 2

        # ReLU block.
        # a_i(x) = W_i x + b_i.
        self.W = torch.tensor(
            [
                [1.0, 0.3],     # degenerate at x0 because b = 0
                [-0.4, 1.0],    # active at x0 because b = 0.5
                [0.7, -0.2],    # inactive at x0 because b = -0.4
            ],
            dtype=dtype,
            device=device,
        )

        self.b = torch.tensor(
            [0.0, 0.5, -0.4],
            dtype=dtype,
            device=device,
        )

        self.c = torch.tensor(
            [1.2, 0.8, 1.5],
            dtype=dtype,
            device=device,
        )

        # Free affine passthrough.
        self.v = torch.tensor(
            [0.1, -0.2],
            dtype=dtype,
            device=device,
        )

        self.b0 = torch.tensor(
            0.0,
            dtype=dtype,
            device=device,
        )

        # Quadratic block.
        self.alpha = torch.tensor(
            0.7,
            dtype=dtype,
            device=device,
        )

        self.B = torch.tensor(
            [
                [0.6, -0.2],
                [0.1, 0.5],
            ],
            dtype=dtype,
            device=device,
        )

        self.e = torch.tensor(
            [0.3, -0.1],
            dtype=dtype,
            device=device,
        )

        # Conic/norm block.
        # u(x) = A x + d.
        # At x0=0, u(x0)=0.
        self.lam = torch.tensor(
            1.3,
            dtype=dtype,
            device=device,
        )

        self.A = torch.eye(
            self.d0,
            dtype=dtype,
            device=device,
        )

        self.d = torch.zeros(
            self.d0,
            dtype=dtype,
            device=device,
        )

        self.x0 = torch.zeros(
            self.d0,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        """
        x: shape (..., d0)
        return: shape (...)
        """
        a = x @ self.W.T + self.b
        relu_part = torch.relu(a) @ self.c

        affine_part = x @ self.v + self.b0

        q = x @ self.B.T + self.e
        quad_part = 0.5 * self.alpha * torch.sum(q * q, dim=-1)

        u = x @ self.A.T + self.d
        norm_part = self.lam * torch.linalg.norm(u, dim=-1)

        return relu_part + affine_part + quad_part + norm_part

    def preactivation(self, x):
        return x @ self.W.T + self.b

    def conic_residual(self, x):
        return x @ self.A.T + self.d

    def deterministic_base_gradient_at_x0(self):
        """
        Gradient contribution fixed across all optimal dual branches at x0.

        Includes:
            - affine passthrough v;
            - strictly active ReLU units;
            - quadratic gradient alpha B^T (B x0 + e);
            - strictly inactive ReLU units contribute zero;
            - degenerate ReLU and degenerate conic parts are excluded.
        """
        x0 = self.x0
        a0 = self.preactivation(x0)

        g = self.v.clone()

        # Strictly active ReLU units.
        active = a0 > 0
        if active.any():
            g = g + self.c[active] @ self.W[active]

        # Quadratic gradient.
        q0 = self.B @ x0 + self.e
        g = g + self.alpha * (self.B.T @ q0)

        return g

    def canonical_gradient_at_x0(self):
        """
        Canonical selector:
            - degenerate ReLU multiplier = 0;
            - zero conic residual selector r = 0.
        """
        return self.deterministic_base_gradient_at_x0()

    def exact_max_readout_directional_derivative(self, direction):
        """
        Compute:
            max_{xi in D*(x0)} G(xi)^T direction.

        At x0:
            - zero ReLU unit i contributes c_i * max(W_i^T d, 0);
            - zero conic residual contributes lambda * ||A d||;
            - all deterministic parts contribute base_grad^T d.
        """
        dvec = direction
        x0 = self.x0
        a0 = self.preactivation(x0)

        base_g = self.deterministic_base_gradient_at_x0()
        value = torch.dot(base_g, dvec)

        # Degenerate ReLU coordinates: a_i(x0)=0.
        deg_relu = torch.abs(a0) <= 1e-14

        for i in torch.where(deg_relu)[0]:
            wi_dot_d = torch.dot(self.W[i], dvec)
            value = value + self.c[i] * torch.clamp(wi_dot_d, min=0.0)

        # Degenerate conic residual u(x0)=0.
        u0 = self.conic_residual(x0)
        if torch.linalg.norm(u0).item() <= 1e-14:
            Ad = self.A @ dvec
            value = value + self.lam * torch.linalg.norm(Ad)

        return value

    def finite_difference_directional_derivative(self, direction, step=1e-7):
        """
        One-sided finite difference:
            [f(x0 + t d) - f(x0)] / t.
        """
        x0 = self.x0
        f0 = self.forward(x0.unsqueeze(0)).squeeze(0)
        f1 = self.forward((x0 + step * direction).unsqueeze(0)).squeeze(0)
        return (f1 - f0) / step

    def random_optimal_dual_readout_at_x0(self, n_samples, seed=0):
        """
        Sample random optimal dual branches at x0 and map them to gradients.

        At x0:
            - active ReLU multiplier is fixed to c_i;
            - inactive ReLU multiplier is fixed to 0;
            - degenerate ReLU multiplier is sampled from [0, c_i];
            - conic residual is zero, so r is sampled from ||r|| <= lambda.
        """
        torch.manual_seed(seed)

        x0 = self.x0
        a0 = self.preactivation(x0)

        base_g = self.deterministic_base_gradient_at_x0()

        grads = []

        for _ in range(n_samples):
            g = base_g.clone()

            # Sample degenerate ReLU multipliers.
            deg_relu = torch.abs(a0) <= 1e-14

            for i in torch.where(deg_relu)[0]:
                nu_i = torch.rand((), dtype=self.dtype, device=self.device) * self.c[i]
                g = g + nu_i * self.W[i]

            # Sample conic multiplier r from Euclidean ball ||r|| <= lambda.
            z = torch.randn(self.d0, dtype=self.dtype, device=self.device)
            z_norm = torch.linalg.norm(z).clamp_min(1e-30)
            z = z / z_norm

            # Uniform radius in 2D ball.
            radius = torch.rand((), dtype=self.dtype, device=self.device).sqrt() * self.lam
            r = radius * z

            g = g + self.A.T @ r

            grads.append(g)

        return torch.stack(grads, dim=0)

    def canonical_dual_norm_at_x0(self):
        """
        Norm of the canonical dual branch.

        We only include the components relevant for comparing sampled branches:
            - ReLU multipliers;
            - quadratic p = alpha q0;
            - conic r.
        """
        x0 = self.x0
        a0 = self.preactivation(x0)

        nu = torch.zeros_like(self.c)
        nu[a0 > 0] = self.c[a0 > 0]
        # degenerate ReLU coordinates remain zero.

        q0 = self.B @ x0 + self.e
        p = self.alpha * q0

        r = torch.zeros(self.d0, dtype=self.dtype, device=self.device)

        return torch.sqrt(torch.sum(nu * nu) + torch.sum(p * p) + torch.sum(r * r))

    def random_dual_norms_at_x0(self, n_samples, seed=0):
        torch.manual_seed(seed)

        x0 = self.x0
        a0 = self.preactivation(x0)

        norms = []

        for _ in range(n_samples):
            nu = torch.zeros_like(self.c)

            # Active ReLU multipliers fixed.
            nu[a0 > 0] = self.c[a0 > 0]

            # Degenerate ReLU multipliers sampled from [0, c_i].
            deg_relu = torch.abs(a0) <= 1e-14
            for i in torch.where(deg_relu)[0]:
                nu[i] = torch.rand((), dtype=self.dtype, device=self.device) * self.c[i]

            # Quadratic p fixed.
            q0 = self.B @ x0 + self.e
            p = self.alpha * q0

            # Conic r sampled from ||r|| <= lambda.
            z = torch.randn(self.d0, dtype=self.dtype, device=self.device)
            z = z / torch.linalg.norm(z).clamp_min(1e-30)
            radius = torch.rand((), dtype=self.dtype, device=self.device).sqrt() * self.lam
            r = radius * z

            norm_val = torch.sqrt(torch.sum(nu * nu) + torch.sum(p * p) + torch.sum(r * r))
            norms.append(norm_val)

        return torch.stack(norms)


def sample_unit_directions(n, d, dtype, device, seed=0):
    torch.manual_seed(seed)
    z = torch.randn(n, d, dtype=dtype, device=device)
    z = z / torch.linalg.norm(z, dim=1, keepdim=True).clamp_min(1e-30)
    return z


def sample_points(n, d, radius, dtype, device, seed=0):
    torch.manual_seed(seed)
    return radius * torch.randn(n, d, dtype=dtype, device=device)


def run_experiment3(
    n_dirs=1000,
    n_random_dual=5000,
    n_support_points=5000,
    step=1e-7,
    device="cpu",
):
    dtype = torch.float64
    model = DegenerateSOCICNN(dtype=dtype, device=device)

    t0 = time.perf_counter()

    x0 = model.x0
    f0 = model.forward(x0.unsqueeze(0)).squeeze(0)

    a0 = model.preactivation(x0)
    u0 = model.conic_residual(x0)

    # --------------------------------------------------------
    # Part A: directional derivative check
    # --------------------------------------------------------
    dirs = sample_unit_directions(
        n=n_dirs,
        d=model.d0,
        dtype=dtype,
        device=device,
        seed=123,
    )

    exact_vals = []
    fd_vals = []
    canonical_vals = []

    g_can = model.canonical_gradient_at_x0()

    for i in range(n_dirs):
        dvec = dirs[i]

        exact_val = model.exact_max_readout_directional_derivative(dvec)
        fd_val = model.finite_difference_directional_derivative(dvec, step=step)
        can_val = torch.dot(g_can, dvec)

        exact_vals.append(exact_val)
        fd_vals.append(fd_val)
        canonical_vals.append(can_val)

    exact_vals = torch.stack(exact_vals)
    fd_vals = torch.stack(fd_vals)
    canonical_vals = torch.stack(canonical_vals)

    fd_abs_err = torch.abs(fd_vals - exact_vals)

    # How often canonical selector is strictly below the true directional derivative.
    can_gap = exact_vals - canonical_vals
    canonical_strict_gap_frac = (can_gap > 1e-8).to(dtype).mean()

    # --------------------------------------------------------
    # Part B: random dual slopes never exceed max readout
    # --------------------------------------------------------
    random_grads = model.random_optimal_dual_readout_at_x0(
        n_samples=n_random_dual,
        seed=456,
    )

    # To avoid an enormous matrix, sample a subset of directions if needed.
    # slopes[j, i] = random_grads[j]^T dirs[i]
    slopes = random_grads @ dirs.T
    exact_matrix = exact_vals.unsqueeze(0).expand_as(slopes)

    excess = slopes - exact_matrix
    max_excess = torch.max(excess)
    mean_max_gap = torch.mean(exact_matrix - slopes)

    # --------------------------------------------------------
    # Part C: canonical subgradient support inequality
    # --------------------------------------------------------
    ys = sample_points(
        n=n_support_points,
        d=model.d0,
        radius=2.0,
        dtype=dtype,
        device=device,
        seed=789,
    )

    f_y = model.forward(ys)
    affine_support = f0 + ys @ g_can

    margins = f_y - affine_support
    min_support_margin = torch.min(margins)
    mean_support_margin = torch.mean(margins)

    # --------------------------------------------------------
    # Part D: canonical branch is minimum-norm among sampled duals
    # --------------------------------------------------------
    can_dual_norm = model.canonical_dual_norm_at_x0()
    random_dual_norms = model.random_dual_norms_at_x0(
        n_samples=n_random_dual,
        seed=999,
    )

    min_random_dual_norm = torch.min(random_dual_norms)
    mean_random_dual_norm = torch.mean(random_dual_norms)
    norm_excess_min = min_random_dual_norm - can_dual_norm

    runtime_ms = 1000.0 * (time.perf_counter() - t0)

    result = {
        "a0": a0.detach().cpu(),
        "u0_norm": torch.linalg.norm(u0).item(),
        "n_dirs": n_dirs,
        "n_random_dual": n_random_dual,
        "n_support_points": n_support_points,
        "fd_abs_err_mean": fd_abs_err.mean().item(),
        "fd_abs_err_max": fd_abs_err.max().item(),
        "canonical_gap_mean": can_gap.mean().item(),
        "canonical_gap_max": can_gap.max().item(),
        "canonical_strict_gap_frac": canonical_strict_gap_frac.item(),
        "max_random_dual_excess": max_excess.item(),
        "mean_max_gap_random_dual": mean_max_gap.item(),
        "min_support_margin": min_support_margin.item(),
        "mean_support_margin": mean_support_margin.item(),
        "canonical_dual_norm": can_dual_norm.item(),
        "min_random_dual_norm": min_random_dual_norm.item(),
        "mean_random_dual_norm": mean_random_dual_norm.item(),
        "norm_excess_min": norm_excess_min.item(),
        "runtime_ms": runtime_ms,
    }

    return result


def print_result(res):
    print("\nExperiment 3: Degeneracy diagnostic")
    print("=" * 88)

    print("Degenerate construction")
    print("-" * 88)
    print(f"ReLU preactivations at x0 : {res['a0'].numpy()}")
    print(f"Conic residual norm at x0 : {res['u0_norm']:.3e}")
    print()

    print("A. Directional derivative = max dual readout")
    print("-" * 88)
    print(f"Directions                         : {res['n_dirs']}")
    print(f"Finite-diff abs err mean           : {res['fd_abs_err_mean']:.3e}")
    print(f"Finite-diff abs err max            : {res['fd_abs_err_max']:.3e}")
    print(f"Canonical gap mean                 : {res['canonical_gap_mean']:.3e}")
    print(f"Canonical gap max                  : {res['canonical_gap_max']:.3e}")
    print(f"Frac. canonical strictly below max : {res['canonical_strict_gap_frac']:.4f}")
    print()

    print("B. Random optimal dual readouts are bounded by max readout")
    print("-" * 88)
    print(f"Random dual samples                : {res['n_random_dual']}")
    print(f"Max random-dual excess             : {res['max_random_dual_excess']:.3e}")
    print(f"Mean max-minus-random gap          : {res['mean_max_gap_random_dual']:.3e}")
    print()

    print("C. Canonical selector is a valid subgradient")
    print("-" * 88)
    print(f"Support test points                : {res['n_support_points']}")
    print(f"Minimum support margin             : {res['min_support_margin']:.3e}")
    print(f"Mean support margin                : {res['mean_support_margin']:.3e}")
    print()

    print("D. Canonical selector is minimum-norm among sampled dual branches")
    print("-" * 88)
    print(f"Canonical dual norm                : {res['canonical_dual_norm']:.6e}")
    print(f"Minimum sampled dual norm          : {res['min_random_dual_norm']:.6e}")
    print(f"Mean sampled dual norm             : {res['mean_random_dual_norm']:.6e}")
    print(f"Min sampled norm - canonical norm  : {res['norm_excess_min']:.3e}")
    print()

    print(f"Runtime                            : {res['runtime_ms']:.2f} ms")
    print("=" * 88)

    print("\nLaTeX-style summary row:")
    print(
        f"{res['n_dirs']} & "
        f"${res['fd_abs_err_mean']:.2e}$ & "
        f"${res['fd_abs_err_max']:.2e}$ & "
        f"{res['canonical_strict_gap_frac']:.3f} & "
        f"${res['max_random_dual_excess']:.2e}$ & "
        f"${res['min_support_margin']:.2e}$ \\\\"
    )


if __name__ == "__main__":
    # CPU is enough.
    device = "cpu"

    res = run_experiment3(
        n_dirs=1000,
        n_random_dual=5000,
        n_support_points=5000,
        step=1e-7,
        device=device,
    )

    print_result(res)