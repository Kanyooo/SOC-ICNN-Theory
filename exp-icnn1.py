import time
import math
import torch


# ============================================================
# Experiment 1: Exact first-order readout for SOC-ICNN
# ============================================================
# Verify:
#
#   g_readout(x)
#   =
#   v
#   + sum_l W_l^T nu_l
#   + sum_h alpha_h B_h^T (B_h x + e_h)
#   + sum_g A_g^T r_g
#
# matches
#
#   g_autodiff(x) = d f_SOC(x) / dx
#
# on nondegenerate samples.
# ============================================================


class RandomSOCICNN:
    def __init__(
        self,
        d0=20,
        width=64,
        depth=4,
        n_quad=2,
        n_norm=2,
        quad_dim=20,
        norm_dim=20,
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
                # U_1 z_0 is not used because z_0 = 0.
                self.U.append(None)
            else:
                # Nonnegative hidden-to-hidden weights.
                self.U.append(scale * torch.rand(width, width, dtype=dtype, device=device))

            self.b.append(scale * torch.randn(width, dtype=dtype, device=device))

        # Nonnegative final output weights.
        self.c = torch.rand(width, dtype=dtype, device=device) + 0.1

        # Free affine passthrough.
        self.v = scale * torch.randn(d0, dtype=dtype, device=device)
        self.b0 = torch.randn((), dtype=dtype, device=device)

        # Quadratic blocks:
        # alpha_h / 2 * ||B_h x + e_h||^2
        self.B = []
        self.e = []
        self.alpha = []

        for _ in range(n_quad):
            self.B.append(scale * torch.randn(quad_dim, d0, dtype=dtype, device=device))
            self.e.append(scale * torch.randn(quad_dim, dtype=dtype, device=device))
            self.alpha.append(torch.rand((), dtype=dtype, device=device) + 0.2)

        # Norm/conic blocks:
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

        Returns
        -------
        f: shape (N,)
        cache: dict containing preactivations, hidden states, q_h, u_g.
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

    def canonical_readout_gradient(self, x, eps=1e-12):
        """
        Compute canonical dual readout gradient manually.

        Formula:
            g_can(x)
            =
            v
            + sum_l W_l^T nu_l
            + sum_h alpha_h B_h^T (B_h x + e_h)
            + sum_g A_g^T r_g

        with
            nu_L = c * 1_{a_L > 0}
            nu_l = (U_{l+1}^T nu_{l+1}) * 1_{a_l > 0}

        and
            r_g = lambda_g * u_g / ||u_g||, if u_g != 0
                  0, otherwise.
        """
        _, cache = self.forward_with_cache(x)
        a_list = cache["a_list"]
        q_list = cache["q_list"]
        u_list = cache["u_list"]

        N = x.shape[0]

        # ---------- ReLU canonical multipliers ----------
        nu_list = [None for _ in range(self.depth)]

        # Last layer:
        # nu_L = c * 1_{a_L > 0}
        nu = self.c.unsqueeze(0).expand(N, -1) * (a_list[-1] > 0).to(x.dtype)
        nu_list[-1] = nu

        # Backward recursion:
        # nu_l = (U_{l+1}^T nu_{l+1}) * 1_{a_l > 0}
        #
        # In matrix form:
        # forward: a_{l+1} = W_{l+1} x + U_{l+1} z_l + b_{l+1}
        # U_{l+1}: shape (width_next, width_current)
        # batch recursion: nu_l = nu_{l+1} @ U_{l+1}
        for l in range(self.depth - 2, -1, -1):
            upper = nu_list[l + 1] @ self.U[l + 1]
            nu_l = upper * (a_list[l] > 0).to(x.dtype)
            nu_list[l] = nu_l

        # ---------- readout gradient ----------
        grad = self.v.unsqueeze(0).expand(N, -1).clone()

        # ReLU part: sum_l W_l^T nu_l
        # batch version: nu_l @ W_l
        for l in range(self.depth):
            grad = grad + nu_list[l] @ self.W[l]

        # Quadratic part: alpha_h B_h^T q_h
        # batch version: alpha_h * q_h @ B_h
        for h in range(self.n_quad):
            grad = grad + self.alpha[h] * (q_list[h] @ self.B[h])

        # Norm/conic part: A_g^T r_g
        # r_g = lambda_g u_g / ||u_g||
        for g in range(self.n_norm):
            u = u_list[g]
            norm_u = torch.linalg.norm(u, dim=1, keepdim=True)

            # canonical selector: if norm is numerically zero, choose r=0
            r = torch.where(
                norm_u > eps,
                self.lam[g] * u / norm_u.clamp_min(eps),
                torch.zeros_like(u),
            )
            grad = grad + r @ self.A[g]

        return grad, cache

    def nondegenerate_mask(self, cache, tol=1e-10):
        """
        Nondegenerate condition:
            all ReLU preactivations are nonzero,
            all conic residuals u_g are nonzero.
        """
        a_list = cache["a_list"]
        u_list = cache["u_list"]

        N = a_list[0].shape[0]
        mask = torch.ones(N, dtype=torch.bool, device=a_list[0].device)

        for a in a_list:
            mask = mask & (torch.min(torch.abs(a), dim=1).values > tol)

        for u in u_list:
            mask = mask & (torch.linalg.norm(u, dim=1) > tol)

        return mask


def run_experiment(
    trials=250,
    d0=20,
    width=64,
    depth=4,
    n_quad=2,
    n_norm=2,
    quad_dim=20,
    norm_dim=20,
    seed=0,
    sample_radius=1.0,
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

    # Random test inputs.
    torch.manual_seed(seed + 123)
    x = sample_radius * torch.randn(trials, d0, dtype=dtype, device=device)
    x.requires_grad_(True)

    # -------------------------
    # Autodiff gradient
    # -------------------------
    t0 = time.perf_counter()

    f, cache = model.forward_with_cache(x)
    loss = f.sum()
    grad_autodiff = torch.autograd.grad(loss, x, create_graph=False)[0]

    # -------------------------
    # Explicit readout gradient
    # -------------------------
    grad_readout, cache2 = model.canonical_readout_gradient(x.detach())

    total_time_ms = 1000.0 * (time.perf_counter() - t0)

    # Nondegenerate filtering.
    mask = model.nondegenerate_mask(cache2)

    retained = int(mask.sum().item())
    retained_rate = retained / trials

    if retained == 0:
        raise RuntimeError(
            "No nondegenerate samples retained. Try reducing tol or changing seed."
        )

    ga = grad_autodiff.detach()[mask]
    gr = grad_readout.detach()[mask]

    diff = gr - ga

    l2_err_each = torch.linalg.norm(diff, dim=1)
    rel_err_each = l2_err_each / torch.linalg.norm(ga, dim=1).clamp_min(1e-30)

    cos_each = torch.sum(gr * ga, dim=1) / (
        torch.linalg.norm(gr, dim=1).clamp_min(1e-30)
        * torch.linalg.norm(ga, dim=1).clamp_min(1e-30)
    )

    result = {
        "trials": trials,
        "retained": retained,
        "retained_rate": retained_rate,
        "grad_l2_err_mean": l2_err_each.mean().item(),
        "grad_l2_err_max": l2_err_each.max().item(),
        "grad_rel_err_mean": rel_err_each.mean().item(),
        "grad_rel_err_max": rel_err_each.max().item(),
        "cosine_mean": cos_each.mean().item(),
        "cosine_min": cos_each.min().item(),
        "runtime_ms": total_time_ms,
    }

    return result


def print_result(result):
    print("\nExperiment 1: Exact first-order readout")
    print("=" * 72)
    print(f"Trials          : {result['trials']}")
    print(f"Retained        : {result['retained']}")
    print(f"Retained Rate   : {result['retained_rate']:.4f}")
    print(f"Grad L2 Err     : mean={result['grad_l2_err_mean']:.3e}, "
          f"max={result['grad_l2_err_max']:.3e}")
    print(f"Grad Rel. Err   : mean={result['grad_rel_err_mean']:.3e}, "
          f"max={result['grad_rel_err_max']:.3e}")
    print(f"Cosine Sim.     : mean={result['cosine_mean']:.12f}, "
          f"min={result['cosine_min']:.12f}")
    print(f"Runtime         : {result['runtime_ms']:.3f} ms")
    print("=" * 72)

    print("\nLaTeX-style table row:")
    print(
        f"{result['trials']} & "
        f"{result['retained_rate']:.4f} & "
        f"${result['grad_l2_err_mean']:.2e}$ & "
        f"${result['grad_rel_err_mean']:.2e}$ & "
        f"{result['cosine_mean']:.12f} & "
        f"{result['runtime_ms']:.2f} \\\\"
    )


if __name__ == "__main__":
    # CPU is enough. Use CUDA only if you want:
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    result = run_experiment(
        trials=250,
        d0=20,
        width=64,
        depth=4,
        n_quad=2,
        n_norm=2,
        quad_dim=20,
        norm_dim=20,
        seed=0,
        sample_radius=1.0,
        device=device,
    )

    print_result(result)