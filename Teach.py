#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment 4: White-box SOC-ICNN inference comparison.

We solve the downstream inference problem

    min_x F_y(x) = f_SOC(x) + beta/2 ||x - y||^2

and compare five methods:

    1) WhiteBox-GD:
       explicit dual readout gradient + gradient descent.

    2) WhiteBox-Newton:
       explicit dual readout gradient + explicit local Hessian + damped Newton.

    3) Torch-GD:
       autograd gradient + gradient descent.

    4) Torch-Newton:
       autograd gradient + autograd Hessian + damped Newton.

    5) Torch-LBFGS-Ref:
       high-accuracy numerical reference for computing GapToRef.

The goal is to demonstrate the full white-box inference pipeline:
    forward pass -> dual multiplier readout -> gradient/Hessian construction
    -> first-/second-order inference -> diagnostics.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    input_dim: int = 10
    hidden_dim: int = 32
    depth: int = 3

    n_quad: int = 1
    quad_dim: int = 8
    n_norm: int = 2
    norm_dim: int = 8

    beta: float = 10.0
    seed: int = 2026
    n_queries: int = 30

    max_gd_iter: int = 800
    max_newton_iter: int = 50
    max_lbfgs_iter: int = 300

    tol: float = 1e-8
    armijo_c: float = 1e-4
    max_backtrack: int = 30

    gd_init_step: float = 0.1
    newton_init_step: float = 1.0
    newton_damping: float = 1e-10

    out_detail_csv: str = "exp4_whitebox_inference_compare_detail.csv"
    out_summary_csv: str = "exp4_whitebox_inference_compare_summary.csv"


# ============================================================
# SOC-ICNN model
# ============================================================
class SmallSOCICNN(nn.Module):
    """
    Small but nontrivial SOC-ICNN:

        f(x) = ReLU-ICNN(x)
             + sum_h alpha_h/2 ||B_h x + e_h||^2
             + sum_g lambda_g ||A_g x + d_g||_2.

    Every hidden layer has a direct affine x-passthrough term W_l x.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        d = cfg.input_dim
        m = cfg.hidden_dim
        L = cfg.depth

        scale_w = 0.7 / math.sqrt(d)
        scale_u = 0.8 / max(1, m)

        self.W = nn.ParameterList()
        self.U = nn.ParameterList()
        self.b = nn.ParameterList()

        for ell in range(L):
            self.W.append(nn.Parameter(torch.randn(m, d) * scale_w))
            self.b.append(nn.Parameter(0.05 * torch.randn(m)))

            if ell == 0:
                self.U.append(nn.Parameter(torch.zeros(m, m), requires_grad=False))
            else:
                self.U.append(nn.Parameter(torch.empty(m, m).uniform_(0.0, scale_u)))

        self.c_raw = nn.Parameter(torch.randn(m))
        self.v = nn.Parameter(torch.randn(d) * scale_w)
        self.b0 = nn.Parameter(torch.zeros(()))

        self.B = nn.ParameterList()
        self.e = nn.ParameterList()
        self.alpha_raw = nn.ParameterList()

        for _ in range(cfg.n_quad):
            self.B.append(nn.Parameter(torch.randn(cfg.quad_dim, d) * scale_w))
            self.e.append(nn.Parameter(0.05 * torch.randn(cfg.quad_dim)))
            self.alpha_raw.append(nn.Parameter(torch.tensor(0.8)))

        self.A = nn.ParameterList()
        self.d = nn.ParameterList()
        self.lambda_raw = nn.ParameterList()

        for _ in range(cfg.n_norm):
            self.A.append(nn.Parameter(torch.randn(cfg.norm_dim, d) * scale_w))
            self.d.append(nn.Parameter(0.05 * torch.randn(cfg.norm_dim)))
            self.lambda_raw.append(nn.Parameter(torch.tensor(0.6)))

    def c(self) -> torch.Tensor:
        return torch.abs(self.c_raw)

    def alphas(self) -> List[torch.Tensor]:
        return [torch.abs(a) + 1e-6 for a in self.alpha_raw]

    def lambdas(self) -> List[torch.Tensor]:
        return [torch.abs(l) + 1e-6 for l in self.lambda_raw]

    def forward_with_cache(self, x: torch.Tensor) -> Dict[str, object]:
        zs = []
        preacts = []
        masks = []

        z_prev = None

        for ell in range(self.cfg.depth):
            a = self.W[ell] @ x + self.b[ell]

            if ell > 0:
                a = a + self.U[ell] @ z_prev

            z = torch.relu(a)

            zs.append(z)
            preacts.append(a)
            masks.append((a > 0).to(x.dtype))

            z_prev = z

        relu_val = torch.dot(self.c(), zs[-1]) + torch.dot(self.v, x) + self.b0

        qs = []
        quad_val = torch.zeros((), dtype=x.dtype)

        for B, e, alpha in zip(self.B, self.e, self.alphas()):
            q = B @ x + e
            qs.append(q)
            quad_val = quad_val + 0.5 * alpha * torch.dot(q, q)

        us = []
        norm_val = torch.zeros((), dtype=x.dtype)

        for A, d, lam in zip(self.A, self.d, self.lambdas()):
            u = A @ x + d
            us.append(u)
            norm_val = norm_val + lam * torch.linalg.norm(u)

        return {
            "value": relu_val + quad_val + norm_val,
            "zs": zs,
            "preacts": preacts,
            "masks": masks,
            "qs": qs,
            "us": us,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_cache(x)["value"]

    def canonical_dual_and_readout(self, x: torch.Tensor) -> Dict[str, object]:
        """
        Explicit white-box readout:
            - ReLU multipliers nu_l
            - quadratic multipliers p_h = alpha_h q_h
            - conic multipliers r_g = lambda_g u_g / ||u_g||
            - gradient readout
            - local Hessian formula
        """
        cache = self.forward_with_cache(x)
        masks = cache["masks"]
        L = self.cfg.depth

        # ReLU multipliers.
        nus = [torch.zeros_like(masks[0]) for _ in range(L)]
        nus[-1] = self.c() * masks[-1]

        for ell in range(L - 2, -1, -1):
            upper = self.U[ell + 1].T @ nus[ell + 1]
            nus[ell] = upper * masks[ell]

        # Quadratic multipliers.
        ps = []
        for alpha, q in zip(self.alphas(), cache["qs"]):
            ps.append(alpha * q)

        # Conic multipliers.
        rs = []
        for lam, u in zip(self.lambdas(), cache["us"]):
            unorm = torch.linalg.norm(u)
            if float(unorm.detach()) > 1e-12:
                rs.append(lam * u / unorm)
            else:
                rs.append(torch.zeros_like(u))

        # Gradient readout for f_SOC.
        grad_f = self.v.clone()

        for ell in range(L):
            grad_f = grad_f + self.W[ell].T @ nus[ell]

        for B, p in zip(self.B, ps):
            grad_f = grad_f + B.T @ p

        for A, r in zip(self.A, rs):
            grad_f = grad_f + A.T @ r

        # Local Hessian for f_SOC.
        H = torch.zeros(self.cfg.input_dim, self.cfg.input_dim, dtype=x.dtype)

        for B, alpha in zip(self.B, self.alphas()):
            H = H + alpha * (B.T @ B)

        for A, lam, u in zip(self.A, self.d, cache["us"]):
            pass

        # Correct conic Hessian loop.
        for A, lam, u in zip(self.A, self.lambdas(), cache["us"]):
            unorm = torch.linalg.norm(u)
            if float(unorm.detach()) > 1e-10:
                uhat = u / unorm
                P = torch.eye(u.numel(), dtype=x.dtype) - torch.outer(uhat, uhat)
                H = H + lam * A.T @ (P / unorm) @ A

        min_abs_preact = min(
            float(torch.min(torch.abs(a)).detach())
            for a in cache["preacts"]
        )

        if len(cache["us"]) > 0:
            min_norm_u = min(
                float(torch.linalg.norm(u).detach())
                for u in cache["us"]
            )
        else:
            min_norm_u = float("inf")

        return {
            "cache": cache,
            "nus": nus,
            "ps": ps,
            "rs": rs,
            "grad_f": grad_f,
            "hess_f": H,
            "min_abs_preact": min_abs_preact,
            "min_norm_u": min_norm_u,
        }


# ============================================================
# Objective and derivatives
# ============================================================
def objective(model: SmallSOCICNN, x: torch.Tensor, y: torch.Tensor, beta: float) -> torch.Tensor:
    return model(x) + 0.5 * beta * torch.sum((x - y) ** 2)


def whitebox_grad_hess(
    model: SmallSOCICNN,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    need_hess: bool = True,
) -> Dict[str, object]:
    out = model.canonical_dual_and_readout(x)

    grad = out["grad_f"] + cfg.beta * (x - y)

    if need_hess:
        H = out["hess_f"] + cfg.beta * torch.eye(cfg.input_dim, dtype=x.dtype)
    else:
        H = None

    return {
        "grad": grad,
        "hess": H,
        "min_abs_preact": out["min_abs_preact"],
        "min_norm_u": out["min_norm_u"],
    }


def torch_grad_hess(
    model: SmallSOCICNN,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    need_hess: bool = True,
) -> Dict[str, object]:
    x_var = x.clone().detach().requires_grad_(True)
    val = objective(model, x_var, y, cfg.beta)

    grad = torch.autograd.grad(val, x_var, create_graph=False)[0].detach()

    if need_hess:
        def fun(z: torch.Tensor) -> torch.Tensor:
            return objective(model, z, y, cfg.beta)

        H = torch.autograd.functional.hessian(fun, x.clone().detach()).detach()
    else:
        H = None

    with torch.no_grad():
        out = model.canonical_dual_and_readout(x.detach())

    return {
        "grad": grad,
        "hess": H,
        "min_abs_preact": out["min_abs_preact"],
        "min_norm_u": out["min_norm_u"],
    }


# ============================================================
# Line search
# ============================================================
def armijo_line_search(
    model: SmallSOCICNN,
    x: torch.Tensor,
    y: torch.Tensor,
    direction: torch.Tensor,
    grad: torch.Tensor,
    cfg: Config,
    init_step: float,
) -> Dict[str, object]:
    f0 = objective(model, x, y, cfg.beta)
    descent = float(torch.dot(grad, direction).detach())

    eta = init_step
    n_backtrack = 0

    # If direction is not descent, force a steepest-descent fallback.
    if descent >= 0.0:
        direction = -grad
        descent = -float(torch.dot(grad, grad).detach())
        eta = min(init_step, cfg.gd_init_step)

    for _ in range(cfg.max_backtrack):
        x_new = x + eta * direction
        f_new = objective(model, x_new, y, cfg.beta)

        if float(f_new.detach()) <= float(f0.detach()) + cfg.armijo_c * eta * descent:
            return {
                "x_new": x_new.detach(),
                "step": eta,
                "n_backtrack": n_backtrack,
                "accepted": True,
            }

        eta *= 0.5
        n_backtrack += 1

    # Very conservative fallback.
    return {
        "x_new": (x - 1e-4 * grad).detach(),
        "step": 1e-4,
        "n_backtrack": n_backtrack,
        "accepted": False,
    }


# ============================================================
# Solvers
# ============================================================
def run_gradient_descent(
    model: SmallSOCICNN,
    y: torch.Tensor,
    cfg: Config,
    mode: str,
) -> Dict[str, object]:
    assert mode in {"whitebox", "torch"}

    x = y.clone().detach()
    total_backtracks = 0

    t0 = time.perf_counter()

    for it in range(cfg.max_gd_iter):
        if mode == "whitebox":
            der = whitebox_grad_hess(model, x, y, cfg, need_hess=False)
            method_name = "WhiteBox-GD"
        else:
            der = torch_grad_hess(model, x, y, cfg, need_hess=False)
            method_name = "Torch-GD"

        grad = der["grad"]
        grad_norm = float(torch.linalg.norm(grad).detach())

        if grad_norm <= cfg.tol:
            break

        direction = -grad

        ls = armijo_line_search(
            model=model,
            x=x,
            y=y,
            direction=direction,
            grad=grad,
            cfg=cfg,
            init_step=cfg.gd_init_step,
        )

        x = ls["x_new"]
        total_backtracks += ls["n_backtrack"]

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    if mode == "whitebox":
        final_der = whitebox_grad_hess(model, x, y, cfg, need_hess=False)
        method_name = "WhiteBox-GD"
    else:
        final_der = torch_grad_hess(model, x, y, cfg, need_hess=False)
        method_name = "Torch-GD"

    final_grad_norm = float(torch.linalg.norm(final_der["grad"]).detach())

    return {
        "Method": method_name,
        "x": x,
        "Obj": float(objective(model, x, y, cfg.beta).detach()),
        "GradNorm": final_grad_norm,
        "Iters": it + 1,
        "RuntimeMs": runtime_ms,
        "Backtracks": total_backtracks,
        "MinAbsPreactivation": final_der["min_abs_preact"],
        "MinNormConicResidual": final_der["min_norm_u"],
    }


def run_newton(
    model: SmallSOCICNN,
    y: torch.Tensor,
    cfg: Config,
    mode: str,
) -> Dict[str, object]:
    assert mode in {"whitebox", "torch"}

    x = y.clone().detach()
    I = torch.eye(cfg.input_dim, dtype=x.dtype)
    total_backtracks = 0

    t0 = time.perf_counter()

    for it in range(cfg.max_newton_iter):
        if mode == "whitebox":
            der = whitebox_grad_hess(model, x, y, cfg, need_hess=True)
            method_name = "WhiteBox-Newton"
        else:
            der = torch_grad_hess(model, x, y, cfg, need_hess=True)
            method_name = "Torch-Newton"

        grad = der["grad"]
        H = der["hess"]

        grad_norm = float(torch.linalg.norm(grad).detach())

        if grad_norm <= cfg.tol:
            break

        try:
            direction = torch.linalg.solve(H + cfg.newton_damping * I, -grad)
        except RuntimeError:
            direction = -grad

        ls = armijo_line_search(
            model=model,
            x=x,
            y=y,
            direction=direction,
            grad=grad,
            cfg=cfg,
            init_step=cfg.newton_init_step,
        )

        x = ls["x_new"]
        total_backtracks += ls["n_backtrack"]

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    if mode == "whitebox":
        final_der = whitebox_grad_hess(model, x, y, cfg, need_hess=True)
        method_name = "WhiteBox-Newton"
    else:
        final_der = torch_grad_hess(model, x, y, cfg, need_hess=True)
        method_name = "Torch-Newton"

    final_grad_norm = float(torch.linalg.norm(final_der["grad"]).detach())

    return {
        "Method": method_name,
        "x": x,
        "Obj": float(objective(model, x, y, cfg.beta).detach()),
        "GradNorm": final_grad_norm,
        "Iters": it + 1,
        "RuntimeMs": runtime_ms,
        "Backtracks": total_backtracks,
        "MinAbsPreactivation": final_der["min_abs_preact"],
        "MinNormConicResidual": final_der["min_norm_u"],
    }


def run_torch_lbfgs_reference(
    model: SmallSOCICNN,
    y: torch.Tensor,
    cfg: Config,
) -> Dict[str, object]:
    x = y.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [x],
        lr=1.0,
        max_iter=cfg.max_lbfgs_iter,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    n_eval = 0
    t0 = time.perf_counter()

    def closure():
        nonlocal n_eval
        optimizer.zero_grad()
        val = objective(model, x, y, cfg.beta)
        val.backward()
        n_eval += 1
        return val

    try:
        optimizer.step(closure)
    except RuntimeError:
        # In rare nonsmooth cases LBFGS may fail. We still return its current point.
        pass

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    x_final = x.detach()

    der = torch_grad_hess(model, x_final, y, cfg, need_hess=False)
    final_grad_norm = float(torch.linalg.norm(der["grad"]).detach())

    return {
        "Method": "Torch-LBFGS-Ref",
        "x": x_final,
        "Obj": float(objective(model, x_final, y, cfg.beta).detach()),
        "GradNorm": final_grad_norm,
        "Iters": n_eval,
        "RuntimeMs": runtime_ms,
        "Backtracks": 0,
        "MinAbsPreactivation": der["min_abs_preact"],
        "MinNormConicResidual": der["min_norm_u"],
    }


# ============================================================
# Diagnostics
# ============================================================
def readout_consistency_at_point(
    model: SmallSOCICNN,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
) -> Dict[str, float]:
    """
    Check explicit white-box gradient/Hessian against torch autograd at the same point.
    """
    wb = whitebox_grad_hess(model, x, y, cfg, need_hess=True)
    tg = torch_grad_hess(model, x, y, cfg, need_hess=True)

    grad_err = float(torch.linalg.norm(wb["grad"] - tg["grad"]).detach())
    grad_rel = grad_err / max(float(torch.linalg.norm(tg["grad"]).detach()), 1e-12)

    hess_err = float(torch.linalg.norm(wb["hess"] - tg["hess"]).detach())
    hess_rel = hess_err / max(float(torch.linalg.norm(tg["hess"]).detach()), 1e-12)

    return {
        "ReadoutGradErr": grad_err,
        "ReadoutGradRelErr": grad_rel,
        "ReadoutHessErr": hess_err,
        "ReadoutHessRelErr": hess_rel,
    }


# ============================================================
# Main
# ============================================================
def main() -> None:
    cfg = Config()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = SmallSOCICNN(cfg)

    rows = []

    print("=" * 100)
    print("Experiment 4 | White-box vs torch inference comparison")
    print("=" * 100)
    print(
        f"d={cfg.input_dim}, hidden={cfg.hidden_dim}, depth={cfg.depth}, "
        f"quad={cfg.n_quad}x{cfg.quad_dim}, norm={cfg.n_norm}x{cfg.norm_dim}, "
        f"beta={cfg.beta}, queries={cfg.n_queries}"
    )

    for qid in range(cfg.n_queries):
        torch.manual_seed(cfg.seed + 1000 + qid)
        y = torch.randn(cfg.input_dim)

        results = []

        results.append(run_gradient_descent(model, y, cfg, mode="whitebox"))
        results.append(run_newton(model, y, cfg, mode="whitebox"))
        results.append(run_gradient_descent(model, y, cfg, mode="torch"))
        results.append(run_newton(model, y, cfg, mode="torch"))
        results.append(run_torch_lbfgs_reference(model, y, cfg))

        # Numerical reference:
        # Use the best value among all solvers, including LBFGS.
        # This is more robust than trusting one method blindly near nonsmooth boundaries.
        ref_obj = min(r["Obj"] for r in results)

        # Readout diagnostic at the WhiteBox-Newton solution.
        wb_newton_x = [r for r in results if r["Method"] == "WhiteBox-Newton"][0]["x"]
        diag = readout_consistency_at_point(model, wb_newton_x, y, cfg)

        for r in results:
            row = {
                "Query": qid,
                "Method": r["Method"],
                "Obj": r["Obj"],
                "GapToRef": r["Obj"] - ref_obj,
                "GradNorm": r["GradNorm"],
                "Iters": r["Iters"],
                "RuntimeMs": r["RuntimeMs"],
                "Backtracks": r["Backtracks"],
                "MinAbsPreactivation": r["MinAbsPreactivation"],
                "MinNormConicResidual": r["MinNormConicResidual"],
                **diag,
            }
            rows.append(row)

        print(
            f"[Query {qid:02d}] "
            f"best={ref_obj:.6e} | "
            + " | ".join(
                f"{r['Method']}: gap={r['Obj'] - ref_obj:.2e}, "
                f"gn={r['GradNorm']:.2e}, it={r['Iters']}"
                for r in results
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_detail_csv, index=False)

    summary = (
        df.groupby("Method")
        .agg(
            GapMean=("GapToRef", "mean"),
            GapMax=("GapToRef", "max"),
            ObjMean=("Obj", "mean"),
            GradNormMean=("GradNorm", "mean"),
            GradNormMedian=("GradNorm", "median"),
            ItersMean=("Iters", "mean"),
            RuntimeMsMean=("RuntimeMs", "mean"),
            RuntimeMsStd=("RuntimeMs", "std"),
            BacktracksMean=("Backtracks", "mean"),
            ReadoutGradErrMean=("ReadoutGradErr", "mean"),
            ReadoutGradRelErrMean=("ReadoutGradRelErr", "mean"),
            ReadoutHessErrMean=("ReadoutHessErr", "mean"),
            ReadoutHessRelErrMean=("ReadoutHessRelErr", "mean"),
            MinAbsPreactivationMean=("MinAbsPreactivation", "mean"),
            MinNormConicResidualMean=("MinNormConicResidual", "mean"),
        )
        .reset_index()
    )

    # Put methods in a stable, meaningful order.
    order = {
        "WhiteBox-GD": 0,
        "WhiteBox-Newton": 1,
        "Torch-GD": 2,
        "Torch-Newton": 3,
        "Torch-LBFGS-Ref": 4,
    }
    summary["Order"] = summary["Method"].map(order)
    summary = summary.sort_values("Order").drop(columns=["Order"])

    summary.to_csv(cfg.out_summary_csv, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nSaved files:")
    print(cfg.out_detail_csv)
    print(cfg.out_summary_csv)


if __name__ == "__main__":
    main()