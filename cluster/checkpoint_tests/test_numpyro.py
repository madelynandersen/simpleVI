"""Can a NumPyro SVI fit be stopped at an arbitrary iteration and resumed exactly?

Model: 1-D Gaussian, known variance (100 observations), AutoNormal guide,
Adam(5e-4), 1 MC sample -- the "default" setup of numpyro_runner.py.

Production path: modulars.numpyro_rr_test.run_restart_1d (one jitted fori_loop).
Resumable path:  the same SVI object, run in jitted chunks; the full SVIState
                 (params, Adam moments + step count, PRNG key) is saved after
                 every chunk and restored in a new process.

Run with the num_pyro environment:  python cluster/checkpoint_tests/test_numpyro.py
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.optim import Adam

import ckpt_harness  # also puts the repo root on sys.path
from modulars.numpyro_rr_test import run_restart_1d, _prepare_model_args
from modulars import gaussian_1d

SEED = 3
STEP_SIZE = 5e-4
MAX_CHUNK = 1000  # largest chunk run_chunk accepts (buffer size)
DATA = gaussian_1d(100, 3.0, 1.0, seed=0)


def model(y):
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    numpyro.sample("obs", dist.Normal(mu, 1.0), obs=y)


def production(n_iters):
    loc, scale = run_restart_1d(SEED, model, DATA, "mu", n_iters, track_start=1, run_multi=False,
                                step_size=STEP_SIZE)
    return np.stack([np.asarray(loc), np.asarray(scale)], axis=1)


class Resumable:
    def __init__(self):
        self.svi = SVI(model, AutoNormal(model), Adam(step_size=STEP_SIZE), loss=Trace_ELBO())
        self.args = _prepare_model_args(data=DATA)
        svi, args = self.svi, self.args

        # production calls svi.init inside its jit, so do the same here
        self._init = jax.jit(lambda key: svi.init(key, *args))

        # m is a *traced* argument and the buffer has a fixed size, so this compiles ONCE and
        # every chunk length runs the identical XLA program. (With m static, each chunk length
        # compiled to a different program and results differed by 1 float32 ulp.)
        @jax.jit
        def chunk(state, m):
            def body(i, carry):
                state, buf = carry
                state, _ = svi.update(state, *args)
                p = svi.get_params(state)
                buf = buf.at[i].set(jnp.stack([p["mu_auto_loc"], p["mu_auto_scale"]]))
                return state, buf
            return jax.lax.fori_loop(0, m, body, (state, jnp.zeros((MAX_CHUNK, 2))))
        self._chunk = chunk
        self.state = None

    def init(self):
        self.state = self._init(jax.random.PRNGKey(SEED))

    def run_chunk(self, m):
        assert 0 < m <= MAX_CHUNK
        self.state, buf = self._chunk(self.state, jnp.int32(m))
        return np.asarray(buf)[:m]

    def get_state(self):
        # leaves only (plain numpy arrays); the tree structure comes from a fresh init on restore
        return [np.asarray(x) for x in jax.tree_util.tree_leaves(jax.device_get(self.state))]

    def set_state(self, leaves):
        self.init()  # builds the guide and gives the tree structure
        ref_leaves, treedef = jax.tree_util.tree_flatten(self.state)

        def restore(ref, x):
            # keep JAX's "weak type" flag (Adam's step counter is a weakly typed int32): a strongly
            # typed leaf changes the jitted program the resumed run compiles to
            if ref.weak_type and np.ndim(x) == 0:
                out = jnp.asarray(x.item())
            else:
                out = jnp.asarray(x, dtype=ref.dtype)
            assert out.dtype == ref.dtype and out.weak_type == ref.weak_type and out.shape == ref.shape
            return out

        self.state = jax.tree_util.tree_unflatten(treedef, [restore(r, x) for r, x in zip(ref_leaves, leaves)])


if __name__ == "__main__":
    print("numpyro", numpyro.__version__, "| jax", jax.__version__)
    ckpt_harness.main(__file__, "numpyro", {"default_1mc": (production, Resumable)})
