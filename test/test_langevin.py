import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from diffrax import diffeqsolve, LangevinTerm, SaveAt, VirtualBrownianTree


def _solvers():
    # solver, order
    yield diffrax.ALIGN(0.1)
    yield diffrax.ShARK()
    yield diffrax.SRA1()
    yield diffrax.SEA()
    yield diffrax.SORT(0.01)
    yield diffrax.ShOULD(0.01)


@pytest.mark.parametrize("solver", _solvers())
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
@pytest.mark.parametrize("dim", [1, 3])
def test_shape(solver, dtype, dim):
    if dtype == jnp.float16 and isinstance(solver, (diffrax.SORT, diffrax.ShOULD)):
        pytest.skip(
            "Due to the use of multivariate nomal in the the computation"
            " of space-time-time Levy area, SORT and ShOULD are not"
            " compatible with float16"
        )
    t0, t1 = 0.3, 1.0
    saveat = SaveAt(ts=jnp.linspace(t0, t1, 10, dtype=dtype))
    u = dtype(1.0)
    gam = dtype(1.0)
    vec_u = jnp.ones((dim,), dtype=dtype)
    vec_gam = jnp.ones((dim,), dtype=dtype)
    x0 = jnp.zeros((dim,), dtype=dtype)
    v0 = jnp.zeros((dim,), dtype=dtype)
    y0 = (x0, v0)
    f = lambda x: 0.5 * x
    shp_dtype = jax.ShapeDtypeStruct((dim,), dtype)
    bm = VirtualBrownianTree(
        t0,
        t1,
        tol=2**-6,
        shape=shp_dtype,
        key=jrandom.PRNGKey(4),
        levy_area=solver.minimal_levy_area(),
    )
    for args in [
        (gam, u, f),
        (vec_gam, u, f),
        (gam, vec_u, f),
        (vec_gam, vec_u, f),
    ]:
        terms = LangevinTerm(args, bm)
        sol = diffeqsolve(
            terms, solver, t0, t1, dt0=0.3, y0=y0, args=None, saveat=saveat
        )
        assert sol.ys is not None
        for entry in sol.ys:
            assert entry.shape == (10, dim)
            assert jnp.dtype(entry) == dtype
