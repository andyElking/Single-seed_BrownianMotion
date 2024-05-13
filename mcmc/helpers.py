import dataclasses
from typing import Callable, Optional

import diffrax
import equinox as eqx
import jax
import jax.random as jr
import jax.tree_util as jtu
import optimistix as optx
from diffrax import (
    AbstractBrownianPath,
    AbstractTerm,
    ControlTerm,
    LangevinTerm,
    LevyArea,
    MultiTerm,
    ODETerm,
    VirtualBrownianTree,
)
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, Shaped


all_ode_solvers = (
    diffrax.Bosh3(),
    diffrax.Dopri5(),
    diffrax.Dopri8(),
    diffrax.Euler(),
    diffrax.Ralston(),
    diffrax.Midpoint(),
    diffrax.Heun(),
    diffrax.LeapfrogMidpoint(),
    diffrax.ReversibleHeun(),
    diffrax.Tsit5(),
    diffrax.ImplicitEuler(),
    diffrax.Kvaerno3(),
    diffrax.Kvaerno4(),
    diffrax.Kvaerno5(),
)

all_split_solvers = (
    diffrax.Sil3(),
    diffrax.KenCarp3(),
    diffrax.KenCarp4(),
    diffrax.KenCarp5(),
)


def implicit_tol(solver):
    if isinstance(solver, diffrax.AbstractImplicitSolver):
        return eqx.tree_at(
            lambda s: (s.root_finder.rtol, s.root_finder.atol, s.root_finder.norm),
            solver,
            (1e-3, 1e-6, optx.rms_norm),
        )
    return solver


def random_pytree(key, treedef, dtype):
    keys = jr.split(key, treedef.num_leaves)
    leaves = []
    for key in keys:
        dimkey, sizekey, valuekey = jr.split(key, 3)
        num_dims = jr.randint(dimkey, (), 0, 5).item()
        dim_sizes = jr.randint(sizekey, (num_dims,), 0, 5)
        value = jr.normal(valuekey, tuple(dim_sizes.tolist()), dtype=dtype)
        leaves.append(value)
    return jtu.tree_unflatten(treedef, leaves)


treedefs = [
    jtu.tree_structure(x)
    for x in (
        0,
        None,
        {"a": [0, 0], "b": 0},
    )
]


def _no_nan(x):
    if eqx.is_array(x):
        return x.at[jnp.isnan(x)].set(8.9568)  # arbitrary magic value
    else:
        return x


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8, equal_nan=False):
    if equal_nan:
        x = jtu.tree_map(_no_nan, x)
        y = jtu.tree_map(_no_nan, y)
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def path_l2_dist(
    ys1: PyTree[Shaped[Array, "repeats times ?*channels"], " T"],
    ys2: PyTree[Shaped[Array, "repeats times ?*channels"], " T"],
):
    # first compute the square of the difference and sum over
    # all but the first two axes (which represent the number of samples
    # and the length of saveat). Also sum all the PyTree leaves.
    def sum_square_diff(y1, y2):
        square_diff = jnp.square(y1 - y2)
        # sum all but the first two axes
        axes = range(2, square_diff.ndim)
        out = jnp.sum(square_diff, axis=axes)
        return out

    dist = jtu.tree_map(sum_square_diff, ys1, ys2)
    dist = sum(jtu.tree_leaves(dist))  # shape=(num_samples, len(saveat))
    dist = jnp.max(dist, axis=1)  # take sup along the length of integration
    dist = jnp.sqrt(jnp.mean(dist))
    return dist


def get_minimal_la(solver):
    while isinstance(solver, diffrax.HalfSolver):
        solver = solver.solver
    la = getattr(solver, "minimal_levy_area", None)
    if callable(la):
        return la()
    else:
        return ""


def _sde_solve(
    keys: PRNGKeyArray,
    get_terms: Callable[[diffrax.AbstractBrownianPath], diffrax.AbstractTerm],
    w_shape: tuple[int, ...],
    t0: float,
    t1: float,
    y0: PyTree[Array],
    args: PyTree,
    solver: diffrax.AbstractSolver,
    levy_area: Optional[LevyArea],
    dt0: Optional[float],
    controller: Optional[diffrax.AbstractStepSizeController],
    bm_tol: float,
    saveat: diffrax.SaveAt,
):
    if levy_area is None:
        levy_area = get_minimal_la(solver)
    dtype = jnp.result_type(*jtu.tree_leaves(y0))
    struct = jax.ShapeDtypeStruct(w_shape, dtype)
    bm = diffrax.VirtualBrownianTree(
        t0=t0,
        t1=t1,
        shape=struct,
        tol=bm_tol,
        key=keys,
        levy_area=levy_area,
    )
    terms = get_terms(bm)
    if controller is None:
        controller = diffrax.ConstantStepSize()
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        dt0=dt0,
        y0=y0,
        args=args,
        max_steps=2**17,
        stepsize_controller=controller,
        saveat=saveat,
    )
    steps = sol.stats["num_accepted_steps"]
    if isinstance(solver, diffrax.HalfSolver):
        steps *= 3
    return sol.ys, steps


_batch_sde_solve = eqx.filter_jit(
    eqx.filter_vmap(
        _sde_solve,
        in_axes=(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
)

_batch_sde_solve_multi_y0 = eqx.filter_jit(
    eqx.filter_vmap(
        _sde_solve,
        in_axes=(
            0,
            None,
            None,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
)


@eqx.filter_jit
def sde_solver_strong_order(
    keys: PRNGKeyArray,
    get_terms: Callable[[diffrax.AbstractBrownianPath], diffrax.AbstractTerm],
    w_shape: tuple[int, ...],
    t0: float,
    t1: float,
    y0: PyTree[Array],
    args: PyTree,
    solver: diffrax.AbstractSolver,
    ref_solver: diffrax.AbstractSolver,
    levels: tuple[int, int],
    get_step_controller: Callable[[int], diffrax.AbstractStepSizeController],
    saveat: diffrax.SaveAt,
    bm_tol: float,
):
    levy_area1 = get_minimal_la(solver)
    levy_area2 = get_minimal_la(ref_solver)
    # Stricter levy_area requirements are a longer string, so only override
    # solver's levy_area if the ref_solver requires more levy area
    # TODO: this is a bit hacky, but I'm not sure how else to do it
    levy_area = levy_area1 if len(levy_area1) > len(levy_area2) else levy_area2

    level_coarse, level_fine = levels
    level_ref = 2 + level_fine

    correct_sols, _ = _batch_sde_solve(
        keys,
        get_terms,
        w_shape,
        t0,
        t1,
        y0,
        args,
        ref_solver,
        levy_area,
        None,
        get_step_controller(level_ref),
        bm_tol,
        saveat,
    )

    errs_list, steps_list = [], []
    for level in range(level_coarse, level_fine + 1):
        sols, steps = _batch_sde_solve(
            keys,
            get_terms,
            w_shape,
            t0,
            t1,
            y0,
            args,
            solver,
            levy_area,
            None,
            get_step_controller(level),
            bm_tol,
            saveat,
        )
        errs = path_l2_dist(sols, correct_sols)
        errs_list.append(errs)
        steps_list.append(jnp.mean(steps))
    errs_arr = jnp.array(errs_list)
    steps_arr = jnp.array(steps_list)
    order, _ = jnp.polyfit(jnp.log(1 / steps_arr), jnp.log(errs_arr), 1)
    return steps_arr, errs_arr, order


# TODO: remove this once we have a better way to handle this
# I understand you'd prefer not to have this in the library and
# I agree this is somewhat hacky, but I think passing each of
# these args around separately is a bit of a pain. If this only appeared
# in the tests, I'd be fine with it, but it's also in the examples
# (e.g. srk_example.py) and I'd prefer if it looked a bit cleaner there.
# So how do you recommend we streamline this?
@dataclasses.dataclass
class SDE:
    get_terms: Callable[[AbstractBrownianPath], AbstractTerm]
    args: PyTree
    y0: PyTree[Array]
    t0: float
    t1: float
    w_shape: tuple[int, ...]

    def get_dtype(self):
        return jnp.result_type(*jtu.tree_leaves(self.y0))

    def get_bm(
        self,
        bm_key,
        levy_area: LevyArea,
        tol=2**-14,
    ):
        shp_dtype = jax.ShapeDtypeStruct(self.w_shape, dtype=self.get_dtype())
        return VirtualBrownianTree(self.t0, self.t1, tol, shp_dtype, bm_key, levy_area)


# A more concise function for use in the examples
def simple_sde_order(
    keys,
    sde: SDE,
    solver,
    ref_solver,
    levels,
    get_step_controller,
    saveat,
    bm_tol,
):
    return sde_solver_strong_order(
        keys,
        sde.get_terms,
        sde.w_shape,
        sde.t0,
        sde.t1,
        sde.y0,
        sde.args,
        solver,
        ref_solver,
        levels,
        get_step_controller,
        saveat,
        bm_tol,
    )


def simple_batch_sde_solve(
    keys, sde: SDE, solver, levy_area, dt0, controller, bm_tol, saveat
):
    return _batch_sde_solve(
        keys,
        sde.get_terms,
        sde.w_shape,
        sde.t0,
        sde.t1,
        sde.y0,
        sde.args,
        solver,
        levy_area,
        dt0,
        controller,
        bm_tol,
        saveat,
    )


def get_bqp(t0=0.3, t1=15.0, dtype=jnp.float32):
    grad_f_bqp = lambda x: 4 * x * (jnp.square(x) - 1)
    args_bqp = (dtype(0.8), dtype(0.2), grad_f_bqp)
    y0_bqp = (dtype(0), dtype(0))
    w_shape_bqp = ()

    def get_terms_bqp(bm):
        return LangevinTerm(args_bqp, bm)

    return SDE(get_terms_bqp, None, y0_bqp, t0, t1, w_shape_bqp)


def get_harmonic_oscillator(t0=0.3, t1=15.0, dtype=jnp.float32):
    gamma_hosc = jnp.array([2, 0.5], dtype=dtype)
    u_hosc = jnp.array([0.5, 2], dtype=dtype)
    args_hosc = (gamma_hosc, u_hosc, lambda x: 2 * x)
    x0 = jnp.zeros((2,), dtype=dtype)
    v0 = jnp.zeros((2,), dtype=dtype)
    y0_hosc = (x0, v0)
    w_shape_hosc = (2,)

    def get_terms_hosc(bm):
        return LangevinTerm(args_hosc, bm)

    return SDE(get_terms_hosc, None, y0_hosc, t0, t1, w_shape_hosc)


def get_neals_funnel(t0=0.0, t1=16.0, dtype=jnp.float32):
    def log_p(x):
        z_term = x[0] ** 2 / 6.0
        y_term = jnp.sum(x[1:] ** 2) / jax.lax.stop_gradient(2.0 * jnp.exp(x[0] / 4.0))
        return z_term + y_term

    grad_log_p = jax.grad(log_p)

    gamma = 2.0
    u = 1.0
    args_neal = (gamma, u, grad_log_p)
    y0_neal = (jnp.zeros((10,), dtype=dtype), jnp.zeros((10,), dtype=dtype))
    w_shape_neal = (10,)

    def get_terms_neal(bm):
        return LangevinTerm(args_neal, bm)

    return SDE(get_terms_neal, None, y0_neal, t0, t1, w_shape_neal)


def get_uld3_langevin(t0=0.3, t1=15.0, dtype=jnp.float32):
    # Three particles in 3D space with a potential that has three local minima,
    # at (2, 2, 2), (-2, -2, -2) and (3, -1, 0).
    def single_particle_potential(x):
        assert x.shape == (3,)
        return 1.0 * (
            jnp.sum((x - 2.0 * jnp.ones((3,), dtype=dtype)) ** 2)
            * jnp.sum((x + 2.0 * jnp.ones((3,), dtype=dtype)) ** 2)
            * jnp.sum((x - jnp.array([3, -1, 0], dtype=dtype)) ** 2)
        )

    def potential(x):
        assert x.shape == (9,)
        return (
            single_particle_potential(x[:3])
            + single_particle_potential(x[3:6])
            + single_particle_potential(x[6:])
        )

    grad_potential = jax.grad(potential)

    def single_circ(x):
        assert x.shape == (3,)
        return 0.1 * jnp.array([x[1], -x[0], 0.0])

    def circular_term(x):
        assert x.shape == (9,)
        return jnp.concatenate(
            [
                single_circ(x[:3]),
                single_circ(x[3:6]),
                single_circ(x[6:]),
            ]
        )

    def grad_f(x):
        assert x.shape == (9,)
        # x0 and x1 will do a circular motion, so we will add a term of the form
        force = grad_potential(x) + circular_term(x)
        return 10.0 * force / (jnp.sum(jnp.abs(force)) + 10.0)

    u = 1.0
    gamma = 2.0
    args = (u, gamma, grad_f)
    x0 = jnp.array([-1, 0, 1, 1, 0, -1, 1, 0, -1], dtype=dtype)
    v0 = jnp.zeros((9,), dtype=dtype)
    y0_uld3 = (x0, v0)
    w_shape_uld3 = (9,)

    def get_terms_uld3(bm):
        return LangevinTerm(args, bm)

    return SDE(get_terms_uld3, None, y0_uld3, t0, t1, w_shape_uld3)


def _squareplus(x):
    return 0.5 * (x + jnp.sqrt(x**2 + 4))


def drift(t, y, args):
    mlp, _, _ = args
    return 0.25 * mlp(y)


def diffusion(t, y, args):
    _, mlp, noise_dim = args
    return 1.0 * mlp(y).reshape(3, noise_dim)


def get_mlp_sde(t0, t1, dtype, key, noise_dim):
    driftkey, diffusionkey, ykey = jr.split(key, 3)
    # To Patrick: I had to increase the depth of these MLPs, otherwise many SDE
    # solvers had order ~0.72 which is more than 0.5 + 0.2, which is the maximal
    # tolerated order. I think the issue was that it was too linear and too easy.
    drift_mlp = eqx.nn.MLP(
        in_size=3,
        out_size=3,
        width_size=8,
        depth=2,
        activation=_squareplus,
        final_activation=jnp.tanh,
        key=driftkey,
    )
    diffusion_mlp = eqx.nn.MLP(
        in_size=3,
        out_size=3 * noise_dim,
        width_size=8,
        depth=2,
        activation=_squareplus,
        final_activation=jnp.tanh,
        key=diffusionkey,
    )
    args = (drift_mlp, diffusion_mlp, noise_dim)
    y0 = jr.normal(ykey, (3,), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(drift), ControlTerm(diffusion, bm))

    return SDE(get_terms, args, y0, t0, t1, (noise_dim,))


# This is needed for time_sde (i.e. the additive noise SDE) because initializing
# the weights in the drift MLP with a Gaussian makes the SDE too linear and nice,
# so we need to use a Laplace distribution, which is heavier-tailed.
def lap_init(weight: jax.Array, key) -> jax.Array:
    stddev = 1.0
    return stddev * jax.random.laplace(key, shape=weight.shape, dtype=weight.dtype)


def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)

    def get_weights(model):
        list = []
        for x in jax.tree_util.tree_leaves(model, is_leaf=is_linear):
            if is_linear(x):
                list.extend([x.weight, x.bias])
        return list

    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def get_time_sde(t0, t1, dtype, key, noise_dim):
    y_dim = 7
    driftkey, diffusionkey, ykey = jr.split(key, 3)

    def ft(t):
        return jnp.array(
            [jnp.sin(t), jnp.cos(4 * t), 1.0, 1.0 / (t + 0.5)], dtype=dtype
        )

    drift_mlp = eqx.nn.MLP(
        in_size=y_dim + 4,
        out_size=y_dim,
        width_size=16,
        depth=5,
        activation=_squareplus,
        key=driftkey,
    )

    # The drift weights must be Laplace-distributed,
    # otherwise the SDE is too linear and nice.
    drift_mlp = init_linear_weight(drift_mlp, lap_init, driftkey)

    def _drift(t, y, _):
        return (
            0.01
            * (drift_mlp(jnp.concatenate([y, ft(t)])) - 50.0 * y**3)
            / (jnp.sum(y**2) + 1)
        )

    diffusion_mx = jr.normal(diffusionkey, (4, y_dim, noise_dim), dtype=dtype)

    def _diffusion(t, _, __):
        # This needs a large coefficient to make the SDE not too easy.
        return 1.0 * jnp.tensordot(ft(t), diffusion_mx, axes=1)

    args = (drift_mlp, None, None)
    y0 = jr.normal(ykey, (y_dim,), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(_drift), ControlTerm(_diffusion, bm))

    return SDE(get_terms, args, y0, t0, t1, (noise_dim,))


def get_cir(t0, t1, args, dtype):
    def _drift(t, y, _args):
        a, b, sigma = _args
        # we need to shift b by sigma^2/(4a) to get the Stratonovich form
        tilde_b = b - (sigma**2) / (4 * a)
        return jnp.asarray(a * (tilde_b - y), dtype=dtype)

    def _diffusion(t, y, _args):
        a, b, sigma = _args
        return jnp.asarray(sigma * jnp.sqrt(jax.nn.relu(y)), dtype=dtype)

    def get_terms(bm):
        return MultiTerm(ODETerm(_drift), ControlTerm(_diffusion, bm))

    y0 = jnp.array(1.0, dtype=dtype)
    w_shape = ()
    return SDE(get_terms, args, y0, t0, t1, w_shape)
