import haiku as hk
import jax
import jax.numpy as jnp

@jit
def safe_divide(a, b):
    den = jnp.clip(b, a_min=1e-9) + jnp.clip(b, a_max=1e-9)
    den = den + (den == 0).astype(den.dtype) * 1e-9
    return a / den * (b != 0).astyped(b.dtype)

class Linear(hk.Module):
    def __init__(self, output_size, name=None):
        super().__init__(self, output_size, name)
        self.output_size = output_size

    def __call__(self, x):
        j, k = x.shape[-1], self.output_size
        w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
        w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.ones)
        return jnp.dot(x, w) + b

    @hk.transparent
    def __f(R, w1, w2, x1, x2):
        Z1 = x1 @ w1 # I don't know weather the at notation is supported?
        Z2 = x2 @ w2
        S1 = safe_divide(R, Z1 + Z2)
        S2 = safe_divide(R, Z1 + Z2)

        ## TODO: Fix gradient propogation?
        C1 = x1 * self.gradprop(Z1, x1, S1)[0]
        C2 = x2 * self.gradprop(Z2, x2, S2)[0]

        return C1 + C2

    def rel_prop(self, R, alpha, x, y):
        beta = alpha - 1
        j, k = x.shape[-1], self.output_size
        w = hk.get_parameter("w", [j, k])
        pw = jnp.clip(w, a_min=0)
        nw = jnp.clip(w, a_max=0)
        px = jnp.clip(x, a_min=0)
        nx = jnp.clip(x, a_max=0)

        activator_relevances = self.__f(R, pw, nw, px, nx)
        inhibitor_relevances = self.__f(R, nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

class ReLU(hk.Module):
    def __init__(self, name=None):
        super().__init__(self, name)
    
    def __call__(self, x):
        return jax.nn.relu(x)

    def rel_prop(self, R, alpha, x, y):
        return R