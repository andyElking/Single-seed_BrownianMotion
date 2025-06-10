from .helpers import _batch_sde_solve, _batch_sde_solve_multi_y0, make_underdamped_langevin_term

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu


def run_funnel_mcmc(
    key,
    log_p,
    x0,
    num_particles: int,
    chain_len: int,
    chain_sep: float = 0.1,
    tol: float = 2**-6,
    warmup_mult: float = 32.0,
    warmup_tol_mult: float = 4.0,
    use_adaptive: bool = True,
    solver: diffrax.AbstractSolver = diffrax.QUICSORT(0.1),
):
    key_warmup, key_mcmc = jr.split(key, 2)
    keys_warmup = jr.split(key_warmup, num_particles)
    keys_mcmc = jr.split(key_mcmc, num_particles)
    v0 = jnp.zeros_like(x0)
    y0 = (x0, v0)
    w_shape: tuple[int, ...] = x0.shape

    grad_log_p = jax.jit(jax.grad(log_p))

    def grad_f(x, _):
        return grad_log_p(x)

    gamma, u = 1.0, 1.0

    def get_terms(bm):
        return make_underdamped_langevin_term(gamma, u, grad_f, bm)

    t_warmup = warmup_mult * chain_sep
    tol_warmup = warmup_tol_mult * tol

    if use_adaptive:
        controller_warmup = diffrax.PIDController(
            rtol=0.0, atol=warmup_tol_mult * tol, pcoeff=0.1, icoeff=0.3, dtmin=2**-6
        )
        solver = diffrax.HalfSolver(solver)
    else:
        controller_warmup = diffrax.ConstantStepSize()

    sols_warmup = _batch_sde_solve(
        keys_warmup,
        get_terms,
        w_shape,
        0.0,
        t_warmup,
        y0,
        None,
        solver,
        diffrax.SpaceTimeTimeLevyArea,
        tol_warmup,
        controller_warmup,
        2**-9,
        diffrax.SaveAt(t1=True),
        True,
        True,
    )
    out_warmup = sols_warmup.ys
    y_warm = jtu.tree_map(
        lambda x: jnp.nan_to_num(x[:, 0], nan=0, posinf=0, neginf=0), out_warmup
    )

    t0_mcmc = 4 * chain_sep
    t1_mcmc: float = chain_len * chain_sep + t0_mcmc
    save_ts = jnp.linspace(t0_mcmc, t1_mcmc, num=chain_len, endpoint=True)
    saveat = diffrax.SaveAt(ts=save_ts)
    if use_adaptive:
        dtmin = 2**-10
        bm_tol = dtmin / 2.0
        controller_mcmc = diffrax.PIDController(
            rtol=0.0, atol=tol, pcoeff=0.1, icoeff=0.4, dtmin=dtmin, step_ts=save_ts
        )
    else:
        step_ts = jnp.linspace(0.0, t1_mcmc, num=int(t1_mcmc / tol) + 1)
        step_ts = jnp.unique(jnp.sort(jnp.concatenate((step_ts, save_ts))))
        controller_mcmc = diffrax.StepTo(ts=step_ts)
        bm_tol = tol / 8.0

    sols_mcmc = _batch_sde_solve_multi_y0(
        keys_mcmc,
        get_terms,
        w_shape,
        0.0,
        t1_mcmc,
        y_warm,
        None,
        solver,
        diffrax.SpaceTimeTimeLevyArea,
        None,
        controller_mcmc,
        bm_tol,
        saveat,
        True,
        True,
    )
    out_mcmc = sols_mcmc.ys
    ys_mcmc = jnp.nan_to_num(out_mcmc[0], nan=0, posinf=0, neginf=0)

    steps_warmup = sols_warmup.stats["num_steps"]
    steps_mcmc = sols_mcmc.stats["num_steps"]
    avg_steps_warmup = jnp.mean(steps_warmup)
    avg_steps_mcmc = jnp.mean(steps_mcmc)
    grad_evals_per_sample = (avg_steps_mcmc + avg_steps_warmup) / chain_len
    # When a HalfSolver is used, the number of gradient evaluations is tripled,
    # but the output of batch_sde_solve already accounts for this.

    if isinstance(solver, diffrax.QUICSORT):
        grad_evals_per_sample *= 2

    print(
        f"Steps warmup: {avg_steps_warmup}, steps mcmc: {avg_steps_mcmc},"
        f" gradient evaluations per output: {grad_evals_per_sample}"
    )

    return ys_mcmc, grad_evals_per_sample
