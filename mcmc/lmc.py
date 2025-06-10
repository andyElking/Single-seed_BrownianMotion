import math
from .helpers import (
    _batch_sde_solve_multi_y0,
    make_underdamped_langevin_term,
)
from typing import Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from diffrax import (
    AbstractSolver,
    ConstantStepSize,
    HalfSolver,
    QUICSORT,
    SaveAt,
    SpaceTimeTimeLevyArea,
    StepTo,
)
from jaxtyping import PyTree
from numpyro.infer.util import initialize_model, Predictive


def get_x0(model, model_args, num_particles, key, prior_start):
    if prior_start:
        x0 = Predictive(model, num_samples=num_particles)(key, *model_args)
        x0.pop("obs", None)
        x0.pop("Y", None)
    else:
        model_info = initialize_model(key, model, model_args=model_args)
        x0 = model_info.param_info.z
        x0 = jtu.tree_map(lambda x: jnp.tile(x, (num_particles, 1)), x0)

    # check the jax config if we are using float64 and adjust x0 accordingly
    if jax.config.jax_enable_x64:
        x0 = jtu.tree_map(lambda x: x.astype(jnp.float64), x0)
    else:
        x0 = jtu.tree_map(lambda x: x.astype(jnp.float32), x0)
    return x0


def run_lmc_numpyro(
    key,
    model,
    model_args,
    num_particles: int,
    chain_len: int,
    chain_sep: float,
    dt0: float,
    warmup_len: float,
    warmup_tol_mult: float,
    pid_mcmc: Optional[diffrax.PIDController] = None,
    pid_warmup: Optional[diffrax.PIDController] = None,
    solver: AbstractSolver = QUICSORT(0.1),
    prior_start: bool = False,
):
    model_key, lmc_key = jr.split(key, 2)
    model_info = initialize_model(model_key, model, model_args=model_args)
    log_p = jax.jit(model_info.potential_fn)
    x0 = get_x0(model, model_args, num_particles, model_key, prior_start)

    return run_lmc(
        lmc_key,
        log_p,
        x0,
        num_particles,
        chain_len,
        chain_sep,
        dt0,
        warmup_len,
        warmup_tol_mult,
        pid_mcmc,
        pid_warmup,
        solver,
    )


def run_lmc(
    key,
    log_p,
    x0,
    num_particles: int,
    chain_len: int,
    chain_sep: float,
    dt0: float,
    warmup_len: float,
    warmup_tol_mult: float,
    pid_mcmc: Optional[diffrax.PIDController] = None,
    pid_warmup: Optional[diffrax.PIDController] = None,
    solver: AbstractSolver = QUICSORT(0.1),
):
    key_warmup, key_mcmc = jr.split(key, 2)
    keys_warmup = jr.split(key_warmup, num_particles)
    keys_mcmc = jr.split(key_mcmc, num_particles)
    grad_f = jax.jit(jax.grad(log_p))
    v0 = jtu.tree_map(lambda x: jnp.zeros_like(x), x0)
    y0 = (x0, v0)

    # Number of evaluations per step of solver
    steps_mult = 1
    if isinstance(solver, QUICSORT):
        steps_mult *= 2

    def get_shape(x):
        shape = jnp.shape(x)
        if shape[0] == num_particles:
            return jax.ShapeDtypeStruct(shape[1:], x.dtype)
        return jax.ShapeDtypeStruct(shape, x.dtype)

    w_shape: PyTree[jax.ShapeDtypeStruct] = jtu.tree_map(get_shape, x0)

    gamma, u = 1.0, 1.0

    def get_terms(bm):
        return make_underdamped_langevin_term(gamma, u, grad_f, bm)

    t_warmup = warmup_len * chain_sep
    tol_warmup = warmup_tol_mult * dt0

    warmup2_len = 4

    t0_mcmc = warmup2_len * chain_sep
    t1_mcmc: float = (chain_len - 1) * chain_sep + t0_mcmc
    save_ts = jnp.linspace(t0_mcmc, t1_mcmc, num=chain_len, endpoint=True)
    saveat = SaveAt(ts=save_ts)

    if pid_mcmc is not None:
        assert (pid_mcmc.dtmin is not None) and (pid_mcmc.dtmax is not None)
        if pid_warmup is None:
            _dtmin = pid_mcmc.dtmin * warmup_tol_mult
            _atol = pid_mcmc.atol * warmup_tol_mult
            _rtol = pid_mcmc.rtol * warmup_tol_mult
            _dtmax = min(pid_mcmc.dtmax * warmup_tol_mult, chain_sep)  # type: ignore
            pid_warmup = eqx.tree_at(
                lambda p: [p.dtmin, p.atol, p.rtol, p.dtmax],
                pid_mcmc,
                [_dtmin, _atol, _rtol, _dtmax],
            )
            assert pid_warmup is not None
        bm_tol_warmup = pid_warmup.dtmin / 2.0  # type: ignore
        bm_tol = pid_mcmc.dtmin / 2.0
        controller_warmup = pid_warmup
        controller_mcmc = diffrax.ClipStepSizeController(pid_mcmc, step_ts=save_ts)

        if not isinstance(solver, diffrax.ShARK):
            solver = HalfSolver(solver)
            steps_mult *= 2
    else:
        assert pid_warmup is None
        controller_warmup = ConstantStepSize()
        steps_per_sample = int(math.ceil(chain_sep / dt0))
        num_steps = (chain_len + warmup2_len - 1) * steps_per_sample + 1
        step_ts = jnp.linspace(0.0, t1_mcmc, num=num_steps, endpoint=True)
        num_steps_before_t0 = warmup2_len * steps_per_sample
        save_ts = step_ts[num_steps_before_t0::steps_per_sample]
        assert save_ts.shape == (
            chain_len,
        ), f"{save_ts.shape}, expected {(chain_len,)}"

        controller_mcmc = StepTo(ts=step_ts)
        bm_tol = dt0 / 4.0
        bm_tol_warmup = tol_warmup / 4.0

    sols_warmup = _batch_sde_solve_multi_y0(
        keys_warmup,
        get_terms,
        w_shape,
        0.0,
        t_warmup,
        y0,
        None,
        solver,
        SpaceTimeTimeLevyArea,
        tol_warmup,
        controller_warmup,
        bm_tol_warmup,
        SaveAt(t1=True),
        use_progress_meter=True,
        use_vbt=True,
    )
    out_warmup = sols_warmup.ys
    steps_warmup = sols_warmup.stats["num_steps"]
    y_warm = jtu.tree_map(
        lambda x: jnp.nan_to_num(x[:, 0], nan=0, posinf=0, neginf=0), out_warmup
    )
    # y_warm includes x and v, both are needed for the main chain

    sols_mcmc = _batch_sde_solve_multi_y0(
        keys_mcmc,
        get_terms,
        w_shape,
        0.0,
        t1_mcmc,
        y_warm,
        None,
        solver,
        SpaceTimeTimeLevyArea,
        None,
        controller_mcmc,
        bm_tol,
        saveat,
        use_progress_meter=True,
        use_vbt=True,
    )
    steps_mcmc = sols_mcmc.stats["num_steps"]
    xs_mcmc = sols_mcmc.ys[0]
    xs_mcmc = jtu.tree_map(
        lambda x: jnp.nan_to_num(x, nan=0, posinf=0, neginf=0), xs_mcmc
    )

    avg_steps_warmup = jnp.mean(steps_warmup)
    avg_steps_mcmc = jnp.mean(steps_mcmc)
    grad_evals_per_sample = steps_mult * (avg_steps_mcmc + avg_steps_warmup) / chain_len

    return xs_mcmc, grad_evals_per_sample


# This just runs the chain without warmup
def run_simple_lmc(
    key,
    log_p,
    x0,
    num_particles: int,
    chain_len: int,
    chain_sep: float,
    dt0: float,
    pid: Optional[diffrax.PIDController],
    solver: AbstractSolver,
):
    keys_mcmc = jr.split(key, num_particles)

    grad_log_p = jax.jit(jax.grad(log_p))

    def grad_f(x, _):
        return grad_log_p(x)

    v0 = jtu.tree_map(lambda x: jnp.zeros_like(x), x0)
    y0 = (x0, v0)

    # Number of evaluations per step of solver
    steps_mult = 1
    if isinstance(solver, QUICSORT):
        steps_mult *= 2

    def get_shape(x):
        shape = jnp.shape(x)
        if shape[0] == num_particles:
            return jax.ShapeDtypeStruct(shape[1:], x.dtype)
        return jax.ShapeDtypeStruct(shape, x.dtype)

    w_shape: PyTree[jax.ShapeDtypeStruct] = jtu.tree_map(get_shape, x0)

    gamma, u = 1.0, 1.0

    def get_terms(bm):
        return make_underdamped_langevin_term(gamma, u, grad_f, bm)

    t1_mcmc: float = (chain_len - 1) * chain_sep
    save_ts = jnp.linspace(0.0, t1_mcmc, num=chain_len, endpoint=True)
    saveat = SaveAt(ts=save_ts)

    if pid is not None:
        controller_mcmc = diffrax.ClipStepSizeController(pid, step_ts=save_ts)
        assert pid.dtmin is not None
        bm_tol = pid.dtmin / 2.0

        if not isinstance(solver, diffrax.ShARK):
            solver = HalfSolver(solver)
            steps_mult *= 2
    else:
        steps_per_sample = int(math.ceil(chain_sep / dt0))
        num_steps = (chain_len - 1) * steps_per_sample + 1
        step_ts = jnp.linspace(0.0, t1_mcmc, num=num_steps, endpoint=True)
        save_ts = step_ts[::steps_per_sample]
        assert save_ts.shape == (
            chain_len,
        ), f"{save_ts.shape}, expected {(chain_len,)}"

        controller_mcmc = StepTo(ts=step_ts)
        bm_tol = dt0 / 4.0

    jax.clear_caches()

    sols_mcmc = _batch_sde_solve_multi_y0(
        keys_mcmc,
        get_terms,
        w_shape,
        0.0,
        t1_mcmc,
        y0,
        None,
        solver,
        SpaceTimeTimeLevyArea,
        None,
        controller_mcmc,
        bm_tol,
        saveat,
        use_progress_meter=True,
        use_vbt=True,
    )
    xs_mcmc = sols_mcmc.ys[0]
    xs_mcmc = jtu.tree_map(
        lambda x: jnp.nan_to_num(x, nan=0, posinf=0, neginf=0), xs_mcmc
    )
    avg_accepted = jnp.mean(sols_mcmc.stats["num_accepted_steps"])
    avg_rejected = jnp.mean(sols_mcmc.stats["num_rejected_steps"])
    print(
        f"avg accepted: {avg_accepted:.2f},"
        f" avg rejected: {avg_rejected:.2f}"
    )

    cumulative_evals = steps_mult * sols_mcmc.num_steps_running
    assert cumulative_evals.shape == (num_particles, chain_len)
    cumulative_evals = jnp.mean(cumulative_evals, axis=0)

    return xs_mcmc, cumulative_evals, avg_accepted, avg_rejected


def run_simple_lmc_numpyro(
    key,
    model,
    model_args,
    num_particles: int,
    chain_len: int,
    chain_sep: float,
    dt0: float,
    pid: Optional[diffrax.PIDController] = None,
    solver: AbstractSolver = QUICSORT(0.1),
    prior_start: bool = False,
):
    model_key, lmc_key = jr.split(key, 2)
    model_info = initialize_model(model_key, model, model_args=model_args)
    log_p = jax.jit(model_info.potential_fn)
    x0 = get_x0(model, model_args, num_particles, model_key, prior_start)
    return run_simple_lmc(
        lmc_key,
        log_p,
        x0,
        num_particles,
        chain_len,
        chain_sep,
        dt0,
        pid,
        solver,
    )
