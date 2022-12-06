import haiku as hk
import jax
import jax.numpy as jnp

@jax.jit
def safe_divide(a, b):
    den = jnp.clip(b, a_min=1e-9) + jnp.clip(b, a_max=1e-9)
    den = den + (den == 0).astype(den.dtype) * 1e-9
    return a / den * (b != 0).astype(b.dtype)

class Linear(hk.Module):
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size
    def __call__(self, x):
        j, k = x.shape[-1], self.output_size
        w_init = hk.initializers.TruncatedNormal(1. / jnp.sqrt(j))
        w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.ones)
        return jnp.dot(x, w) + b
    def rel_prop(self, R, x, alpha=1):
        beta = alpha - 1
        j, k = x.shape[-1], self.output_size
        w = hk.get_parameter("w", [j, k], init=jnp.zeros)
        pw = jnp.clip(w, a_min=0)
        nw = jnp.clip(w, a_max=0)
        px = jnp.clip(x, a_min=0)
        nx = jnp.clip(x, a_max=0)

        @hk.transparent
        def f(R, w1, w2, x1, x2):
            z1, vjp_x1 = jax.vjp(lambda x: jnp.dot(x, w1), x1)
            z2, vjp_x2 = jax.vjp(lambda x: jnp.dot(x, w2), x2)
            s1 = safe_divide(R, z1 + z2)
            s2 = safe_divide(R, z1 + z2)
            c1 = x1 * vjp_x1(s1)[0]
            c2 = x2 * vjp_x2(s2)[0]

            return c1 + c2

        activator_relevances = f(R, pw, nw, px, nx)
        inhibitor_relevances = f(R, nw, pw, px, nx)
        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R