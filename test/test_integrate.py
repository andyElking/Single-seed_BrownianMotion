import contextlib
import math
import operator
from typing import cast, Literal

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest
import scipy.stats
from equinox.internal import ω
from jaxtyping import Array

from .helpers import (
    all_ode_solvers,
    all_split_solvers,
    get_harmonic_oscillator,
    get_mlp_sde,
    get_time_sde,
    implicit_tol,
    path_l2_dist,
    random_pytree,
    simple_batch_sde_solve,
    simple_sde_order,
    tree_allclose,
    treedefs,
)


def _all_pairs(*args):
    defaults = [arg["default"] for arg in args]
    yield defaults
    for i in range(len(args)):
        for opt in args[i]["opts"]:
            opts = defaults.copy()
            opts[i] = opt
            yield opts
    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            for opt1 in args[i]["opts"]:
                for opt2 in args[j]["opts"]:
                    opts = defaults.copy()
                    opts[i] = opt1
                    opts[j] = opt2
                    yield opts


@pytest.mark.parametrize(
    "solver,t_dtype,y_dtype,treedef,stepsize_controller",
    _all_pairs(
        dict(
            default=diffrax.Euler(),
            opts=(
                diffrax.LeapfrogMidpoint(),
                diffrax.ReversibleHeun(),
                diffrax.Tsit5(),
                diffrax.ImplicitEuler(
                    root_finder=diffrax.VeryChord(rtol=1e-3, atol=1e-6)
                ),
                diffrax.Kvaerno3(root_finder=diffrax.VeryChord(rtol=1e-3, atol=1e-6)),
            ),
        ),
        dict(default=jnp.float32, opts=(int, float, jnp.int32)),
        dict(default=jnp.float32, opts=(jnp.complex64,)),
        dict(default=treedefs[0], opts=treedefs[1:]),
        dict(
            default=diffrax.ConstantStepSize(),
            opts=(diffrax.PIDController(rtol=1e-5, atol=1e-8),),
        ),
    ),
)
def test_basic(solver, t_dtype, y_dtype, treedef, stepsize_controller, getkey):
    if not isinstance(solver, diffrax.AbstractAdaptiveSolver) and isinstance(
        stepsize_controller, diffrax.PIDController
    ):
        return
    if isinstance(
        solver, diffrax.AbstractImplicitSolver
    ) and treedef == jtu.tree_structure(None):
        return

    if jnp.iscomplexobj(y_dtype) and treedef != jtu.tree_structure(None):
        if isinstance(solver, diffrax.AbstractImplicitSolver):
            return
        else:
            complex_warn = pytest.warns(match="Complex dtype")

            def f(t, y, args):
                return jtu.tree_map(lambda yi: -1j * yi, y)
    else:
        complex_warn = contextlib.nullcontext()

        def f(t, y, args):
            return jtu.tree_map(operator.neg, y)

    if t_dtype is int:
        t0 = 0
        t1 = 1
        dt0 = 0.01
    elif t_dtype is float:
        t0 = 0.0
        t1 = 1.0
        dt0 = 0.01
    elif t_dtype is jnp.int32:
        t0 = jnp.array(0, dtype=t_dtype)
        t1 = jnp.array(1, dtype=t_dtype)
        dt0 = jnp.array(0.01, dtype=jnp.float32)
    elif t_dtype is jnp.float32:
        t0 = jnp.array(0, dtype=t_dtype)
        t1 = jnp.array(1, dtype=t_dtype)
        dt0 = jnp.array(0.01, dtype=t_dtype)
    else:
        raise ValueError
    y0 = random_pytree(getkey(), treedef, dtype=y_dtype)
    try:
        with complex_warn:
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f),
                solver,
                t0,
                t1,
                dt0,
                y0,
                stepsize_controller=stepsize_controller,
            )
    except Exception as e:
        if isinstance(stepsize_controller, diffrax.ConstantStepSize) and str(
            e
        ).startswith("Nonlinear solve diverged"):
            # Implicit method failed to converge. A very normal thing to have happen;
            # usually we'd use adaptive timestepping to handle it.
            pass
        else:
            raise
    else:
        y1 = sol.ys
        # TODO: remove dtype cast, fix Diffrax internals to better respect dtypes.
        if jnp.iscomplexobj(y_dtype):
            true_y1 = jtu.tree_map(
                lambda x, x1: (x * jnp.exp(-1j))[None].astype(x1.dtype), y0, y1
            )
        else:
            true_y1 = jtu.tree_map(
                lambda x, x1: (x * math.exp(-1))[None].astype(x1.dtype), y0, y1
            )
        assert tree_allclose(y1, true_y1, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("solver", all_ode_solvers + all_split_solvers)
def test_ode_order(solver):
    solver = implicit_tol(solver)
    key = jr.PRNGKey(5678)
    akey, ykey = jr.split(key, 2)

    A = jr.normal(akey, (10, 10), dtype=jnp.float64) * 0.5

    if (
        solver.term_structure
        == diffrax.MultiTerm[tuple[diffrax.AbstractTerm, diffrax.AbstractTerm]]
    ):

        def f1(t, y, args):
            return 0.3 * A @ y

        def f2(t, y, args):
            return 0.7 * A @ y

        term = diffrax.MultiTerm(diffrax.ODETerm(f1), diffrax.ODETerm(f2))
    else:

        def f(t, y, args):
            return A @ y

        term = diffrax.ODETerm(f)
    t0 = 0
    t1 = 4
    y0 = jr.normal(ykey, (10,), dtype=jnp.float64)

    true_yT = jax.scipy.linalg.expm((t1 - t0) * A) @ y0
    exponents = []
    errors = []
    for exponent in [0, -1, -2, -3, -4, -6, -8, -12]:
        dt0 = 2**exponent
        sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, max_steps=None)
        yT = cast(Array, sol.ys)[-1]
        error = jnp.sum(jnp.abs(yT - true_yT))
        if error < 2**-28:
            break
        exponents.append(exponent)
        errors.append(jnp.log2(error))

    order = scipy.stats.linregress(exponents, errors).slope  # pyright: ignore
    # We accept quite a wide range. Improving this test would be nice.
    assert -0.9 < order - solver.order(term) < 0.9


def _solvers():
    # solver, noise, order
    # noise is "any" or "com" or "add" where "com" means commutative and "add" means
    # additive.
    yield diffrax.Euler, "any", 0.5
    yield diffrax.EulerHeun, "any", 0.5
    yield diffrax.Heun, "any", 0.5
    yield diffrax.ItoMilstein, "any", 0.5
    yield diffrax.Midpoint, "any", 0.5
    yield diffrax.ReversibleHeun, "any", 0.5
    yield diffrax.StratonovichMilstein, "any", 0.5
    yield diffrax.SPaRK, "any", 0.5
    yield diffrax.GeneralShARK, "any", 0.5
    yield diffrax.ReversibleHeun, "com", 1
    yield diffrax.StratonovichMilstein, "com", 1
    yield diffrax.SPaRK, "com", 1
    yield diffrax.GeneralShARK, "com", 1
    yield diffrax.SlowRK, "com", 1.5
    yield diffrax.SPaRK, "add", 1.5
    yield diffrax.GeneralShARK, "add", 1.5
    yield diffrax.ShARK, "add", 1.5
    yield diffrax.SRA1, "add", 1.5
    yield diffrax.SEA, "add", 1.0
    yield diffrax.ALIGN, "uld", 2.0
    yield diffrax.ShOULD, "uld", 3.0
    yield diffrax.SORT, "uld", 3.0
    yield diffrax.ShARK, "uld", 2.0


# TODO: For solvers of high order, comparing to Euler or Heun is not good,
# because they are waaaay worse than for example ShARK. ShARK is more precise
# at dt=2**-4 than Euler is at dt=2**-14 (and it takes forever to run at such
# a small dt). Hence , the order of convergence of ShARK seems to plateau at
# discretisations finer than 2**-4.
# I propose the following:
# We can sparate this test into two. First we determine how fast the solver
# converges to its own limit (i.e. using itself as reference), and then
# check whether that limit is the same as the Euler/Heun limit.
# For the second, I would like to make a separate check, where the "correct"
# solution is computed only once and then all solvers are compared to it.
@pytest.mark.parametrize("solver_ctr,noise,theoretical_order", _solvers())
def test_sde_strong_order(
    solver_ctr, noise: Literal["any", "com", "add"], theoretical_order
):
    bmkey = jr.PRNGKey(5678)
    sde_key = jr.PRNGKey(11)
    num_samples = 100
    bmkeys = jr.split(bmkey, num=num_samples)
    t0 = 0.3
    t1 = 5.3

    if noise == "add":
        sde = get_time_sde(t0, t1, jnp.float64, sde_key, noise_dim=7)
    elif noise == "uld":
        sde = get_harmonic_oscillator(t0, t1, jnp.float64)
    else:
        if noise == "com":
            noise_dim = 1
        elif noise == "any":
            noise_dim = 5
        else:
            assert False
        sde = get_mlp_sde(t0, t1, jnp.float64, sde_key, noise_dim=noise_dim)

    if issubclass(solver_ctr, (diffrax.ALIGN, diffrax.ShOULD, diffrax.SORT)):
        solver = solver_ctr(0.01)
    else:
        solver = solver_ctr()
    level_coarse, level_fine = 4, 10

    # We specify the times to which we step in way that each level contains all the
    # steps of the previous level. This is so that we can compare the solutions at
    # all the times in saveat, and not just at the end time.
    def get_step_controller(level):
        step_ts = jnp.linspace(t0, t1, 2**level + 1, endpoint=True)
        return diffrax.StepTo(ts=step_ts)

    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 2**level_coarse + 1, endpoint=True))

    hs, errors, order = simple_sde_order(
        bmkeys,
        sde,
        solver,
        solver,
        (level_coarse, level_fine),
        get_step_controller,
        saveat,
        bm_tol=2**-14,
    )
    # The upper bound needs to be 0.25, otherwise we fail.
    # This still preserves a 0.05 buffer between the intervals
    # corresponding to the different orders.
    print(order)
    assert -0.2 < order - theoretical_order < 0.25


# Make variables to store the correct solutions in.
# This is to avoid recomputing the correct solutions for every solver.
solutions = {
    "Ito": {
        "any": None,
        "com": None,
        "add": None,
        "uld": None,
    },
    "Stratonovich": {
        "any": None,
        "com": None,
        "add": None,
        "uld": None,
    },
}


# Now compare the limit of Euler/Heun to the limit of the other solvers,
# using a single reference solution. We use Euler if the solver is Ito
# and Heun if the solver is Stratonovich.
@pytest.mark.parametrize("solver_ctr,noise,theoretical_order", _solvers())
def test_sde_strong_limit(
    solver_ctr, noise: Literal["any", "com", "add"], theoretical_order
):
    bmkey = jr.PRNGKey(5678)
    sde_key = jr.PRNGKey(11)
    num_samples = 100
    bmkeys = jr.split(bmkey, num=num_samples)
    t0 = 0.3
    t1 = 5.3

    if noise == "add":
        sde = get_time_sde(t0, t1, jnp.float64, sde_key, noise_dim=7)
    elif noise == "uld":
        sde = get_harmonic_oscillator(t0, t1, jnp.float64)
    else:
        if noise == "com":
            noise_dim = 1
        elif noise == "any":
            noise_dim = 5
        else:
            assert False
        sde = get_mlp_sde(t0, t1, jnp.float64, sde_key, noise_dim=noise_dim)

    # Reference solver is always an ODE-viable solver, so its implementation has been
    # verified by the ODE tests like test_ode_order.
    if issubclass(solver_ctr, diffrax.AbstractItoSolver):
        sol_type = "Ito"
        ref_solver = diffrax.Euler()
    elif issubclass(solver_ctr, diffrax.AbstractStratonovichSolver):
        sol_type = "Stratonovich"
        ref_solver = diffrax.Heun()
    else:
        assert False

    if issubclass(solver_ctr, (diffrax.ALIGN, diffrax.ShOULD, diffrax.SORT)):
        solver = solver_ctr(0.01)
    else:
        solver = solver_ctr()

    ts_fine = jnp.linspace(t0, t1, 2**13 + 1, endpoint=True)
    ts_coarse = jnp.linspace(t0, t1, 2**11 + 1, endpoint=True)
    contr_fine = diffrax.StepTo(ts=ts_fine)
    contr_coarse = diffrax.StepTo(ts=ts_coarse)
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t1, 2**6 + 1, endpoint=True))
    levy_area = "space-time"  # must be common for all solvers
    # Skip the test if the solver needs space-time-time Levy area
    if isinstance(solver, (diffrax.SORT, diffrax.ShOULD)):
        pytest.skip("SORT and ShOULD need space-time-time Levy area")

    if solutions[sol_type][noise] is None:
        correct_sol, _ = simple_batch_sde_solve(
            bmkeys, sde, ref_solver, levy_area, None, contr_fine, 2**-14, saveat
        )
        solutions[sol_type][noise] = correct_sol
    else:
        correct_sol = solutions[sol_type][noise]

    sol, _ = simple_batch_sde_solve(
        bmkeys, sde, solver, levy_area, None, contr_coarse, 2**-14, saveat
    )
    error = path_l2_dist(correct_sol, sol)
    print(f"Error: {error}")
    assert error < 0.02


# Step size deliberately chosen not to divide the time interval
@pytest.mark.parametrize(
    "solver_ctr,dt0",
    ((diffrax.Euler, -0.3), (diffrax.Tsit5, -0.3), (diffrax.Tsit5, None)),
)
@pytest.mark.parametrize(
    "saveat",
    (
        diffrax.SaveAt(t0=True),
        diffrax.SaveAt(t1=True),
        diffrax.SaveAt(ts=[3.5, 0.7]),
        diffrax.SaveAt(steps=True),
        diffrax.SaveAt(dense=True),
    ),
)
def test_reverse_time(solver_ctr, dt0, saveat, getkey):
    key = getkey()
    y0 = jr.normal(key, (2, 2))
    stepsize_controller = (
        diffrax.PIDController(rtol=1e-3, atol=1e-6)
        if dt0 is None
        else diffrax.ConstantStepSize()
    )

    def f(t, y, args):
        return -y

    t0 = 4
    t1 = 0.3
    sol1 = diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        solver_ctr(),
        t0,
        t1,
        dt0,
        y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    assert tree_allclose(sol1.t0, jnp.array(4.0))
    assert tree_allclose(sol1.t1, jnp.array(0.3))

    def g(t, y, args):
        return y

    t0 = -4
    t1 = -0.3
    negdt0 = None if dt0 is None else -dt0
    if saveat.subs is not None and saveat.subs.ts is not None:
        saveat = diffrax.SaveAt(ts=[-ti for ti in saveat.subs.ts])
    sol2 = diffrax.diffeqsolve(
        diffrax.ODETerm(g),
        solver_ctr(),
        t0,
        t1,
        negdt0,
        y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )
    assert tree_allclose(sol2.t0, jnp.array(-4.0))
    assert tree_allclose(sol2.t1, jnp.array(-0.3))

    if saveat.subs is not None and (
        saveat.subs.t0
        or saveat.subs.t1
        or saveat.subs.ts is not None
        or saveat.subs.steps
    ):
        assert tree_allclose(sol1.ts, -cast(Array, sol2.ts), equal_nan=True)
        assert tree_allclose(sol1.ys, sol2.ys, equal_nan=True)
    if saveat.dense:
        t = jnp.linspace(0.3, 4, 20)
        for ti in t:
            assert tree_allclose(sol1.evaluate(ti), sol2.evaluate(-ti))
            assert tree_allclose(sol1.derivative(ti), -sol2.derivative(-ti))


def test_semi_implicit_euler():
    term1 = diffrax.ODETerm(lambda t, y, args: -y)
    term2 = diffrax.ODETerm(lambda t, y, args: y)
    y0 = (1.0, -0.5)
    dt0 = 0.00001
    sol1 = diffrax.diffeqsolve(
        (term1, term2),
        diffrax.SemiImplicitEuler(),
        0,
        1,
        dt0,
        y0,
        max_steps=100000,
    )
    term_combined = diffrax.ODETerm(lambda t, y, args: (-y[1], y[0]))
    sol2 = diffrax.diffeqsolve(term_combined, diffrax.Tsit5(), 0, 1, 0.001, y0)
    assert tree_allclose(sol1.ys, sol2.ys)


@pytest.mark.parametrize(
    "solver",
    [
        diffrax.ImplicitEuler(root_finder=diffrax.VeryChord(rtol=1e-3, atol=1e-6)),
        diffrax.Kvaerno5(root_finder=diffrax.VeryChord(rtol=1e-3, atol=1e-6)),
    ],
)
def test_grad_implicit_solve(solver):
    # Check that we work around JAX issue #9374
    # Whilst we're at -- for efficiency -- check the use of PyTree-valued state with
    # implicit solves.

    term = diffrax.ODETerm(lambda t, y, args: (-args * y**ω).ω)

    @jax.jit
    def f(args):
        y0 = (1.0, {"a": 2.0})
        ys = diffrax.diffeqsolve(term, solver, t0=0, t1=1, dt0=0.1, y0=y0, args=args).ys
        ys = cast(Array, ys)
        return jnp.sum(ys[0] + ys[1]["a"])

    grads = jax.jit(jax.grad(f))(1.0)
    assert jnp.isfinite(grads)

    # Test numerical gradients: Diffrax issue #64
    eps = 1e-6
    val = f(1.0)
    val_eps = f(1.0 + eps)
    numerical_grads = (val_eps - val) / eps
    assert tree_allclose(grads, numerical_grads)


def test_concrete_made_jump():
    for constant in (True, False):
        if constant:
            dt0 = 0.1
            stepsize_controller = diffrax.ConstantStepSize()
        else:
            dt0 = None
            stepsize_controller = diffrax.StepTo([0, 0.3, 1])

        @jax.jit
        def run(y0):
            term = diffrax.ODETerm(lambda t, y, args: -y)
            sol = diffrax.diffeqsolve(
                term,
                diffrax.Tsit5(),
                0,
                1,
                dt0,
                y0,
                stepsize_controller=stepsize_controller,
                saveat=diffrax.SaveAt(t1=True, made_jump=True),
                throw=False,
            )
            assert sol.made_jump is False

        run(1)


def test_no_jit():
    # https://github.com/patrick-kidger/diffrax/issues/293
    # https://github.com/patrick-kidger/diffrax/issues/321

    # Test that this doesn't crash.
    with jax.disable_jit():

        def vector_field(t, y, args):
            return jnp.zeros_like(y)

        term = diffrax.ODETerm(vector_field)
        y = jnp.zeros((1,))
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
        diffrax.diffeqsolve(
            term,
            diffrax.Kvaerno4(),
            t0=0,
            t1=1e-2,
            dt0=1e-3,
            stepsize_controller=stepsize_controller,
            y0=y,
        )


def test_static(capfd):
    try:
        diffrax._integrate._PRINT_STATIC = True

        def vector_field(t, y, args):
            return jnp.zeros_like(y)

        term = diffrax.ODETerm(vector_field)
        y = jnp.zeros((1,))
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
        capfd.readouterr()

        diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=0,
            t1=1e-2,
            dt0=1e-3,
            stepsize_controller=stepsize_controller,
            y0=y,
        )
        text, _ = capfd.readouterr()
        assert (
            text == "static_made_jump=False static_result=diffrax._solution.RESULTS<>\n"
        )

        diffrax.diffeqsolve(
            term,
            diffrax.Kvaerno5(),
            t0=0,
            t1=1e-2,
            dt0=1e-3,
            stepsize_controller=stepsize_controller,
            y0=y,
        )
        text, _ = capfd.readouterr()
        assert text == "static_made_jump=False static_result=None\n"
    finally:
        diffrax._integrate._PRINT_STATIC = False


def test_implicit_tol_error():
    msg = "the tolerances for the implicit solver have not been specified"
    with pytest.raises(ValueError, match=msg):
        diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: -y),
            diffrax.Kvaerno5(),
            0,
            1,
            0.01,
            1.0,
        )
