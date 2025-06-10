from mcmc.helpers import SDE, simple_batch_sde_solve, simple_sde_order
from typing import Optional

import diffrax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from diffrax import PIDController
from matplotlib import animation, pyplot as plt  # type: ignore


_markers = [
    "+",
    ",",
    "o",
    "v",
    ".",
    "x",
    "*",
    "^",
    "<",
    ">",
    "h",
    "H",
    "X",
    "D",
    "d",
    "|",
    "_",
]
solvers_3_evals = ["SPaRK"]
solvers_2_evals = ["Heun", "SORT", "ShOULD", "UBU3", "SRA1", "ShARK", "QUICSORT"]


def draw_order_multiple_dict(
    results_dict: dict, title: Optional[str] = None, markers=_markers
):
    fig, ax = plt.subplots()
    fig.set_dpi(200)
    if title is not None:
        ax.set_title(title)

    orders = "Orders:\n"
    scats = []
    for i, (name, result) in enumerate(results_dict.items()):
        steps, errs, _ = result
        if any(solver in name for solver in solvers_3_evals):
            num_evals = 3 * steps
        elif any(solver in name for solver in solvers_2_evals):
            num_evals = 2 * steps
        else:
            num_evals = steps
        trend = np.polyfit(-np.log(num_evals), np.log(errs), 1)
        order, _ = trend
        trend_f = np.poly1d(trend)
        # plot the points
        scat = ax.scatter(
            num_evals, errs, label=f"{name}: {order:.2f}", marker=markers[i]
        )
        scats.append(scat)
        # plot the trend line
        ax.plot(num_evals, np.exp(trend_f(-np.log(num_evals))), linewidth=1.0)
        orders += f"{name}: {order:.2f}\n"
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("RMS error")
    ax.set_xlabel("Average number of vector field evaluations")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin, ymax])
    xmin, xmax = ax.get_xlim()
    ax.set_xlim([xmin / 1.6, xmax])
    ax.legend(fancybox=True, fontsize="large")
    plt.show()
    return fig


def draw_order_multiple(
    results_list: list, names_list: list, title: Optional[str] = None
):
    results_dict = {name: result for result, name in zip(results_list, names_list)}
    return draw_order_multiple_dict(results_dict, title)


def draw_order_single(results):
    steps, errs, order = results
    plt.plot(steps, errs)
    plt.yscale("log")
    plt.xscale("log")
    pretty_steps = [int(step) for step in steps]
    plt.xticks(ticks=pretty_steps, labels=pretty_steps)
    plt.ylabel("RMS error")
    plt.xlabel("average number of steps")
    plt.show()
    print(f"Order of convergence: {order:.4f}")


def plot_sol_general(sol):
    plt.plot(sol.ts, sol.ys)
    plt.show()


def plot_sol_langevin(sol):
    fig = plt.figure()
    ax = fig.add_subplot()
    x, v = sol.ys
    x = jnp.concat(jtu.tree_leaves(x), axis=-1)
    v = jnp.concat(jtu.tree_leaves(v), axis=-1)
    ax.plot(sol.ts, x, label="x")
    ax.plot(sol.ts, v, label="v")
    # ax.set_ylim([-3.0, 3.0])
    ax.legend()
    plt.show()


def plot_sol3D(sol):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    x1 = sol.ys[0][:, :3]  # this is an Nx3 array which we want to plot
    x2 = sol.ys[0][:, 3:6]
    x3 = sol.ys[0][:, 6:]
    ax.plot(*x1.T, label="particle 1")
    ax.plot(*x2.T, label="particle 2")
    ax.plot(*x3.T, label="particle 3")
    ax.legend()
    plt.show()


FUNNEL_LIMS = [-8, 8]


def animated_funnel_plot(sol, skip=8):
    xs = sol.ys[0]
    num_times = xs.shape[1]
    fig, (ax1, ax2) = plt.subplots(2)
    scat = ax1.scatter(xs[:, 0, 0], xs[:, 0, 1])
    ax1.set(xlim=FUNNEL_LIMS, ylim=FUNNEL_LIMS)

    def update(frame):
        idx = min(frame * skip, num_times - 1)
        x = xs[:, idx, 0]
        y = xs[:, idx, 1]
        scat.set_offsets(jnp.stack([x, y]).T)
        return (scat,)

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=int(num_times / skip) + 2, interval=4
    )

    ax2.scatter(xs[:, -1, 0], xs[:, -1, 1])
    ax2.set(xlim=FUNNEL_LIMS, ylim=FUNNEL_LIMS)
    plt.show()
    return ani, fig, ax1


def draw_funnel(samples):
    samples_rshp = jnp.reshape(samples, (-1, 10))
    fig, ax = plt.subplots(1, 1)
    ax.scatter(samples_rshp[:, 0], samples_rshp[:, 1], alpha=0.2, s=8.0)
    ax.set(xlim=FUNNEL_LIMS, ylim=FUNNEL_LIMS)
    plt.show()


def get_fine_solutions(
    keys, sde: SDE, solver, levy_area, bm_tol, level_coarse, level_fine
):
    saveat = diffrax.SaveAt(
        ts=jnp.linspace(sde.t0, sde.t1, 2**level_coarse + 1, endpoint=True)
    )
    controller = diffrax.StepTo(
        ts=jnp.linspace(sde.t0, sde.t1, 2**level_fine + 1, endpoint=True)
    )
    sol = simple_batch_sde_solve(
        keys, sde, solver, levy_area, None, controller, bm_tol, saveat
    )
    return sol.ys


def save_ref_sol(sol, name):
    filename = f"data/ref_sol_{name}.npy"
    np.save(filename, sol)


def load_ref_sol_langevin(name, allow_pickle=False):
    filename = f"data/ref_sol_{name}.npy"
    sol_x, sol_v = jnp.load(filename, allow_pickle=allow_pickle)
    return sol_x, sol_v


def adjust_saveat_len(ref_sol, target_len):
    shape = jnp.shape(jtu.tree_leaves(ref_sol)[0])
    ref_saveat_len = shape[1]
    adjust_factor = (ref_saveat_len - 1) / (target_len - 1)
    assert int(adjust_factor) == adjust_factor, (
        f"Ref solution saveat is not a multiple of target."
        f" Instead they differ by a factor of {adjust_factor}"
    )
    if adjust_factor > 1:
        ref_sol = jtu.tree_map(lambda x: x[:, :: int(adjust_factor)], ref_sol)
    elif adjust_factor < 1:
        raise ValueError("Ref solution saveat is too coarse")
    return ref_sol


def constant_step_strong_order(
    keys, sde, solver, levels, bm_tol=None, levy_area=None, ref_sol=None
):
    def _step_ts(level):
        return jnp.linspace(sde.t0, sde.t1, 2**level + 1, endpoint=True)

    def get_dt_controller(level):
        return None, diffrax.StepTo(ts=_step_ts(level))

    _saveat = diffrax.SaveAt(ts=_step_ts(levels[0]))
    # make sure the saveat of the ref solution conincides
    if ref_sol is not None:
        ref_sol = adjust_saveat_len(ref_sol, 2 ** levels[0] + 1)
        new_shape = jnp.shape(jtu.tree_leaves(ref_sol)[0])
        assert new_shape[:2] == (len(keys), 2 ** levels[0] + 1)

    if bm_tol is None:
        bm_tol = (sde.t1 - sde.t0) * (2 ** -(levels[1] + 3))
    return simple_sde_order(
        keys,
        sde,
        solver,
        solver,
        levels,
        get_dt_controller,
        _saveat,
        bm_tol,
        levy_area=levy_area,
        ref_solution=ref_sol,
    )  # returns steps, errs, order


def pid_strong_order(
    keys, sde, solver, levels, bm_tol=2**-14, levy_area=None, ref_sol=None
):
    save_ts_pid = jnp.linspace(sde.t0, sde.t1, 65, endpoint=True)

    def get_dt_pid(level):
        pid = PIDController(
            pcoeff=0.1,
            icoeff=0.4,
            rtol=0,
            atol=2**-level,
            dtmin=2**-14,
        )
        return None, diffrax.ClipStepSizeController(
            pid,
            step_ts=save_ts_pid,
        )

    saveat_pid = diffrax.SaveAt(ts=save_ts_pid)
    if ref_sol is not None:
        ref_sol = adjust_saveat_len(ref_sol, len(save_ts_pid))
        new_shape = jnp.shape(jtu.tree_leaves(ref_sol)[0])
        assert new_shape[:2] == (len(keys), len(save_ts_pid))

    steps, errs, order = simple_sde_order(
        keys,
        sde,
        solver,
        solver,
        levels,
        get_dt_pid,
        saveat_pid,
        bm_tol,
        levy_area=levy_area,
        ref_solution=None,
    )

    if isinstance(solver, diffrax.HalfSolver):
        # If the solver is a HalfSolver, then the number of evaluations
        # is doubled
        steps = 3 * steps

    return steps, errs, order


def save_order_results(order_results, name):
    steps, errs, order = order_results
    filename = f"data/order_results_{name}.npy"
    with open(filename, "wb") as f:
        np.save(f, steps)
        np.save(f, errs)
        np.save(f, order)
    print(f"Saved {filename}")


def load_order_results(name):
    filename = f"data/order_results_{name}.npy"
    with open(filename, "rb") as f:
        steps = np.load(f)
        errs = np.load(f)
        order = np.load(f)
    print(f"Loaded {filename}")
    return steps, errs, order


def remove_steps(results, lower, upper):
    steps, errs, order = results
    print(len(steps))
    upper = len(steps) - upper
    steps = steps[lower:upper]
    errs = errs[lower:upper]
    return steps, errs, order
