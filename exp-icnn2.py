import time
import torch


class RandomSOCICNN:
    def __init__(
        self,
        d0=10,
        width=32,
        depth=3,
        n_quad=2,
        n_norm=2,
        quad_dim=10,
        norm_dim=10,
        seed=0,
        dtype=torch.float64,
        device="cpu",
        scale=0.2,
    ):
        torch.manual_seed(seed)

        self.d0 = d0
        self.width = width
        self.depth = depth
        self.n_quad = n_quad
        self.n_norm = n_norm
        self.quad_dim = quad_dim
        self.norm_dim = norm_dim
        self.dtype = dtype
        self.device = device

        # ReLU-ICNN backbone:
        # z_l = ReLU(W_l x + U_l z_{l-1} + b_l)
        # U_l >= 0, c >= 0
        self.W = []
        self.U = []
        self.b = []

        for l in range(depth):
            self.W.append(scale * torch.randn(width, d0, dtype=dtype, device=device))

            if l == 0:
                self.U.append(None)
            else:
                self.U.append(scale * torch.rand(width, width, dtype=dtype, device=device))

            self.b.append(scale * torch.randn(width, dtype=dtype, device=device))

        self.c = torch.rand(width, dtype=dtype, device=device) + 0.1
        self.v = scale * torch.randn(d0, dtype=dtype, device=device)
        self.b0 = torch.randn((), dtype=dtype, device=device)

        # Quadratic branches:
        # alpha_h / 2 * ||B_h x + e_h||^2
        self.B = []
        self.e = []
        self.alpha = []

        for _ in range(n_quad):
            self.B.append(scale * torch.randn(quad_dim, d0, dtype=dtype, device=device))
            self.e.append(scale * torch.randn(quad_dim, dtype=dtype, device=device))
            self.alpha.append(torch.rand((), dtype=dtype, device=device) + 0.2)

        # Norm/conic branches:
        # lambda_g * ||A_g x + d_g||
        self.A = []
        self.d = []
        self.lam = []

        for _ in range(n_norm):
            self.A.append(scale * torch.randn(norm_dim, d0, dtype=dtype, device=device))
            self.d.append(scale * torch.randn(norm_dim, dtype=dtype, device=device))
            self.lam.append(torch.rand((), dtype=dtype, device=device) + 0.2)

    def forward_with_cache(self, x):
        """
        x: shape (N, d0)
        return:
            f: shape (N,)
            cache: preactivations, hidden states, q_h, u_g
        """
        z_list = []
        a_list = []

        z_prev = None

        for l in range(self.depth):
            a = x @ self.W[l].T + self.b[l]

            if l > 0:
                a = a + z_prev @ self.U[l].T

            z = torch.relu(a)

            a_list.append(a)
            z_list.append(z)
            z_prev = z

        f_relu = z_list[-1] @ self.c + x @ self.v + self.b0

        q_list = []
        f_quad = torch.zeros_like(f_relu)

        for h in range(self.n_quad):
            q = x @ self.B[h].T + self.e[h]
            q_list.append(q)
            f_quad = f_quad + 0.5 * self.alpha[h] * torch.sum(q * q, dim=1)

        u_list = []
        f_norm = torch.zeros_like(f_relu)

        for g in range(self.n_norm):
            u = x @ self.A[g].T + self.d[g]
            u_list.append(u)
            f_norm = f_norm + self.lam[g] * torch.linalg.norm(u, dim=1)

        f = f_relu + f_quad + f_norm

        cache = {
            "a_list": a_list,
            "z_list": z_list,
            "q_list": q_list,
            "u_list": u_list,
        }

        return f, cache

    def forward_single(self, x):
        """
        x: shape (d0,)
        return scalar f_SOC(x)
        """
        f, _ = self.forward_with_cache(x.unsqueeze(0))
        return f.squeeze(0)

    def readout_gradient(self, x, eps=1e-12):
        """
        Explicit canonical gradient formula.

        g(x)
        =
        v
        + sum_l W_l^T nu_l
        + sum_h alpha_h B_h^T (B_h x + e_h)
        + sum_g lambda_g A_g^T u_g / ||u_g||

        where the ReLU multipliers are recursively:
            nu_L = c * 1_{a_L > 0}
            nu_l = (U_{l+1}^T nu_{l+1}) * 1_{a_l > 0}.
        """
        _, cache = self.forward_with_cache(x)

        a_list = cache["a_list"]
        q_list = cache["q_list"]
        u_list = cache["u_list"]

        N = x.shape[0]

        # ---------- ReLU canonical multipliers ----------
        nu_list = [None for _ in range(self.depth)]

        nu = self.c.unsqueeze(0).expand(N, -1) * (a_list[-1] > 0).to(x.dtype)
        nu_list[-1] = nu

        for l in range(self.depth - 2, -1, -1):
            upper = nu_list[l + 1] @ self.U[l + 1]
            nu_l = upper * (a_list[l] > 0).to(x.dtype)
            nu_list[l] = nu_l

        # ---------- gradient readout ----------
        grad = self.v.unsqueeze(0).expand(N, -1).clone()

        # ReLU part: sum_l W_l^T nu_l
        for l in range(self.depth):
            grad = grad + nu_list[l] @ self.W[l]

        # Quadratic part: sum_h alpha_h B_h^T q_h
        for h in range(self.n_quad):
            grad = grad + self.alpha[h] * (q_list[h] @ self.B[h])

        # Conic/norm part: sum_g A_g^T r_g
        for g in range(self.n_norm):
            u = u_list[g]
            norm_u = torch.linalg.norm(u, dim=1, keepdim=True)

            r = torch.where(
                norm_u > eps,
                self.lam[g] * u / norm_u.clamp_min(eps),
                torch.zeros_like(u),
            )

            grad = grad + r @ self.A[g]

        return grad

    def formula_hessian_single(self, x, eps=1e-12):
        """
        Explicit local Hessian formula at a nondegenerate point:

        H =
        sum_h alpha_h B_h^T B_h
        +
        sum_g lambda_g A_g^T
              [ (I - uhat_g uhat_g^T) / ||u_g|| ]
              A_g.

        The ReLU branch contributes zero Hessian inside a fixed
        activation region.
        """
        if x.ndim != 1:
            raise ValueError("x must have shape (d0,)")

        H = torch.zeros(self.d0, self.d0, dtype=self.dtype, device=self.device)

        # Quadratic branch Hessian.
        for h in range(self.n_quad):
            H = H + self.alpha[h] * (self.B[h].T @ self.B[h])

        # Norm/conic branch Hessian.
        for g in range(self.n_norm):
            A = self.A[g]
            u = A @ x + self.d[g]
            norm_u = torch.linalg.norm(u)

            if norm_u <= eps:
                raise RuntimeError(
                    f"Conic residual {g} is degenerate: ||u||={norm_u.item():.3e}"
                )

            uhat = u / norm_u
            I = torch.eye(u.shape[0], dtype=self.dtype, device=self.device)

            M = (I - torch.outer(uhat, uhat)) / norm_u

            H = H + self.lam[g] * (A.T @ M @ A)

        return H

    def nondegenerate_single(self, x, tol=1e-10):
        """
        Nondegenerate condition:
            all ReLU preactivations are nonzero,
            all conic residuals are nonzero.
        """
        _, cache = self.forward_with_cache(x.unsqueeze(0))

        for a in cache["a_list"]:
            if torch.min(torch.abs(a)).item() <= tol:
                return False

        for u in cache["u_list"]:
            if torch.linalg.norm(u.squeeze(0)).item() <= tol:
                return False

        return True

    def relu_signatures_single(self, x):
        """
        Return ReLU activation pattern at x.
        """
        _, cache = self.forward_with_cache(x.unsqueeze(0))
        return [(a.squeeze(0) > 0) for a in cache["a_list"]]

    def same_relu_branch_single(self, x, ref_signs):
        """
        Check whether x stays in the same ReLU activation branch.
        """
        signs = self.relu_signatures_single(x)

        for s, r in zip(signs, ref_signs):
            if not torch.equal(s, r):
                return False

        return True


def autodiff_grad_single(model, x):
    x_req = x.detach().clone().requires_grad_(True)
    f = model.forward_single(x_req)
    g = torch.autograd.grad(f, x_req, create_graph=False)[0]
    return g.detach()


def autodiff_hessian_single(model, x):
    x_req = x.detach().clone().requires_grad_(True)
    H = torch.autograd.functional.hessian(lambda z: model.forward_single(z), x_req)
    return H.detach()


def find_nondegenerate_samples(
    model,
    n,
    seed=123,
    sample_radius=1.0,
    tol=1e-10,
    max_tries=100000,
):
    torch.manual_seed(seed)

    xs = []
    tries = 0

    while len(xs) < n and tries < max_tries:
        tries += 1

        x = sample_radius * torch.randn(
            model.d0,
            dtype=model.dtype,
            device=model.device,
        )

        if model.nondegenerate_single(x, tol=tol):
            xs.append(x)

    if len(xs) < n:
        raise RuntimeError(
            f"Only found {len(xs)} nondegenerate samples after {tries} tries."
        )

    return torch.stack(xs, dim=0), tries


def sample_uniform_ball(n, d, radius, dtype, device):
    """
    Uniform samples in Euclidean ball B(0, radius).
    """
    z = torch.randn(n, d, dtype=dtype, device=device)
    z = z / torch.linalg.norm(z, dim=1, keepdim=True).clamp_min(1e-30)

    r = torch.rand(n, 1, dtype=dtype, device=device).pow(1.0 / d) * radius

    return z * r


def experiment2_formula_check(
    trials=100,
    d0=10,
    width=32,
    depth=3,
    n_quad=2,
    n_norm=2,
    quad_dim=10,
    norm_dim=10,
    seed=0,
    device="cpu",
):
    dtype = torch.float64

    model = RandomSOCICNN(
        d0=d0,
        width=width,
        depth=depth,
        n_quad=n_quad,
        n_norm=n_norm,
        quad_dim=quad_dim,
        norm_dim=norm_dim,
        seed=seed,
        dtype=dtype,
        device=device,
    )

    xs, tries = find_nondegenerate_samples(
        model,
        n=trials,
        seed=seed + 11,
        sample_radius=1.0,
    )

    t0 = time.perf_counter()

    grad_l2_err = []
    grad_rel_err = []
    hess_fro_err = []
    hess_rel_err = []
    mineig_formula = []
    mineig_autodiff = []

    for i in range(trials):
        x = xs[i]

        # Gradient: formula vs autodiff.
        g_formula = model.readout_gradient(x.unsqueeze(0)).squeeze(0).detach()
        g_auto = autodiff_grad_single(model, x)

        # Hessian: formula vs autodiff.
        H_formula = model.formula_hessian_single(x).detach()
        H_auto = autodiff_hessian_single(model, x)

        gd = g_formula - g_auto
        Hd = H_formula - H_auto

        grad_l2_err.append(torch.linalg.norm(gd))
        grad_rel_err.append(
            torch.linalg.norm(gd) / torch.linalg.norm(g_auto).clamp_min(1e-30)
        )

        hess_fro_err.append(torch.linalg.norm(Hd, ord="fro"))
        hess_rel_err.append(
            torch.linalg.norm(Hd, ord="fro")
            / torch.linalg.norm(H_auto, ord="fro").clamp_min(1e-30)
        )

        mineig_formula.append(torch.linalg.eigvalsh(H_formula).min())
        mineig_autodiff.append(torch.linalg.eigvalsh(H_auto).min())

    runtime_ms = 1000.0 * (time.perf_counter() - t0)

    def mean(vals):
        return torch.stack(vals).mean().item()

    result = {
        "trials": trials,
        "tries": tries,
        "retained_rate": trials / tries,
        "grad_l2_err_mean": mean(grad_l2_err),
        "grad_rel_err_mean": mean(grad_rel_err),
        "hess_fro_err_mean": mean(hess_fro_err),
        "hess_rel_err_mean": mean(hess_rel_err),
        "mineig_formula_mean": mean(mineig_formula),
        "mineig_autodiff_mean": mean(mineig_autodiff),
        "runtime_ms": runtime_ms,
        "model": model,
        "anchor": xs[0].detach(),
    }

    return result


def experiment2_quadratic_approx(
    model,
    anchor,
    radii=(1e-4, 3e-4, 1e-3),
    n_perturb=500,
    seed=999,
    tol=1e-10,
):
    torch.manual_seed(seed)

    x0 = anchor.detach().clone()

    f0 = model.forward_single(x0).detach()
    g0 = model.readout_gradient(x0.unsqueeze(0)).squeeze(0).detach()
    H0 = model.formula_hessian_single(x0).detach()

    ref_signs = model.relu_signatures_single(x0)

    rows = []

    for radius in radii:
        deltas = sample_uniform_ball(
            n=n_perturb,
            d=model.d0,
            radius=radius,
            dtype=model.dtype,
            device=model.device,
        )

        errors = []
        retained = 0

        for i in range(n_perturb):
            delta = deltas[i]
            x = x0 + delta

            # Retained means:
            # 1. same ReLU activation pattern;
            # 2. all conic residuals remain nonzero.
            same_branch = model.same_relu_branch_single(x, ref_signs)

            _, cache = model.forward_with_cache(x.unsqueeze(0))
            nonzero_conic = True

            for u in cache["u_list"]:
                if torch.linalg.norm(u.squeeze(0)).item() <= tol:
                    nonzero_conic = False
                    break

            if same_branch and nonzero_conic:
                retained += 1

            f_true = model.forward_single(x).detach()
            f_quad = f0 + torch.dot(g0, delta) + 0.5 * (delta @ H0 @ delta)

            errors.append(torch.abs(f_true - f_quad))

        errors = torch.stack(errors)

        rows.append(
            {
                "radius": radius,
                "retained_rate": retained / n_perturb,
                "mean_abs_error": errors.mean().item(),
                "max_abs_error": errors.max().item(),
            }
        )

    return rows


def print_formula_result(res):
    print("\nExperiment 2A: Local gradient/Hessian formula check")
    print("=" * 88)
    print(f"Trials              : {res['trials']}")
    print(f"Sampling tries      : {res['tries']}")
    print(f"Retained rate       : {res['retained_rate']:.4f}")
    print(f"Grad L2 Err         : {res['grad_l2_err_mean']:.3e}")
    print(f"Grad Rel. Err       : {res['grad_rel_err_mean']:.3e}")
    print(f"Hess Fro. Err       : {res['hess_fro_err_mean']:.3e}")
    print(f"Hess Rel. Err       : {res['hess_rel_err_mean']:.3e}")
    print(f"MinEig Formula      : {res['mineig_formula_mean']:.6e}")
    print(f"MinEig Autodiff     : {res['mineig_autodiff_mean']:.6e}")
    print(f"Runtime             : {res['runtime_ms']:.2f} ms")
    print("=" * 88)

    print("\nLaTeX-style table row:")
    print(
        f"{res['trials']} & "
        f"${res['grad_l2_err_mean']:.2e}$ & "
        f"${res['grad_rel_err_mean']:.2e}$ & "
        f"${res['hess_fro_err_mean']:.2e}$ & "
        f"${res['hess_rel_err_mean']:.2e}$ & "
        f"${res['mineig_formula_mean']:.2e}$ & "
        f"${res['mineig_autodiff_mean']:.2e}$ \\\\"
    )


def print_quad_rows(rows):
    print("\nExperiment 2B: Local quadratic approximation")
    print("=" * 72)
    print(
        f"{'Radius':>12} | "
        f"{'Retained Rate':>14} | "
        f"{'Mean Abs Err':>14} | "
        f"{'Max Abs Err':>14}"
    )
    print("-" * 72)

    for row in rows:
        print(
            f"{row['radius']:12.1e} | "
            f"{row['retained_rate']:14.4f} | "
            f"{row['mean_abs_error']:14.3e} | "
            f"{row['max_abs_error']:14.3e}"
        )

    print("=" * 72)

    print("\nLaTeX-style table rows:")
    for row in rows:
        print(
            f"${row['radius']:.0e}$ & "
            f"{row['retained_rate']:.3f} & "
            f"${row['mean_abs_error']:.2e}$ \\\\"
        )


if __name__ == "__main__":
    # Hessian computation is much heavier than gradient computation.
    # Keep d0 moderate unless you have a strong CPU/GPU.
    #
    # If you use CUDA:
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # For this experiment, CPU is usually enough because d0 is small.
    device = "cpu"

    res = experiment2_formula_check(
        trials=100,
        d0=10,
        width=32,
        depth=3,
        n_quad=2,
        n_norm=2,
        quad_dim=10,
        norm_dim=10,
        seed=0,
        device=device,
    )

    print_formula_result(res)

    rows = experiment2_quadratic_approx(
        model=res["model"],
        anchor=res["anchor"],
        radii=(1e-4, 3e-4, 1e-3),
        n_perturb=500,
        seed=999,
    )

    print_quad_rows(rows)