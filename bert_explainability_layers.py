from typing import Callable, Optional, Tuple

import numpy as np

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def safe_divide(a, b):
    den = jnp.clip(b, a_min=1e-9) + jnp.clip(b, a_max=1e-9)
    den = den + (den == 0).astype(den.dtype) * 1e-9
    return a / den * (b != 0).astype(b.dtype)

class RelProp(nn.Module):
    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, cam, *inputs, **kwargs):
        return cam

class RelPropSimple(RelProp):
    def relprop(self, cam, *inputs, **kwargs):
        z, grad_func = jax.vjp(lambda *i : self.__call__(*i, **kwargs), *inputs)
        s = safe_divide(cam, z)
        c = grad_func(s)

        if len(inputs) > 1:
            # We only ever take te relevence propogation with respect to the first two positional arguments
            outputs = []
            outputs.append(inputs[0] * c[0])
            outputs.append(inputs[1] * c[1])
        else:
            outputs = inputs[0] * (c[0])
        return outputs

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class Dropout(nn.Dropout, RelProp):
    pass

class Sofmax(nn.Softmax, RelProp):
    pass

class Tanh(nn.Tanh, RelProp):
    pass

## Verified in jupyter notebook
class MatMul(RelPropSimple):
    def __call__(self, *inputs):
        return jnp.matmul(*inputs)

## Only relies on the RelPropSimple, so if the above works, this should work without verification
class einsum(RelPropSimple):
    equation: str

    def __call__(self, *operands):
        return jnp.einsum(self.equation, *operands)

## Verified in jupyter notebook
class Add(RelPropSimple):
    def __call__(self, *inputs):
        return jnp.add(*inputs)
    
    def relprop(self, cam, *inputs, **kwargs):
        z, grad_func = jax.vjp(lambda *i : self.__call__(*i), *inputs)
        s = safe_divide(cam, z)
        c = grad_func(s)

        a = inputs[0] * c[0]
        b = inputs[1] * c[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(jnp.abs(a_sum), jnp.abs(a_sum) + jnp.abs(b_sum)) * cam.sum()
        b_fact = safe_divide(jnp.abs(b_sum), jnp.abs(a_sum) + jnp.abs(b_sum)) * cam.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs

# Verified in Haiku (not yet in Flax, but I don't know why it would change)
class Dense(fnn.Dense, RelProp):
    def relprop(self, R, *inputs, alpha=1):
        inputs = inputs[0]
        beta = alpha - 1
        j, k = x.shape[-1], self.features
        w = self.variables["params"]["kernel"]
        pw = jnp.clip(w, a_min=0)
        nw = jnp.clip(w, a_max=0)
        px = jnp.clip(x, a_min=0)
        nx = jnp.clip(x, a_max=0)
        
        def __f(R, w1, w2, x1, x2):
            z1, vjp_x1 = jax.vjp(lambda x: jnp.dot(x,w1), x1)
            z2, vjp_x2 = jax.vjp(lambda x: jnp.dot(x,w2), x2)
            s1 = safe_divide(R, z1 + z2)
            s2 = safe_divide(R, z1 + z2)

            c1 = x1 * vjp_x1(s1)[0]
            c2 = x2 * vjp_x2(s2)[0]

            return c1 + c2

        activator_relevances = __f(R, pw, nw, px, nx)
        inhibitor_relevances = __f(R, nw, pw, px, nx)
        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R

## Verified in notebook
class IndexSelect(RelProp):
    def __call__(self, inputs, dim, indices):
        return jnp.take(inputs, indices, axis=dim)

    def relprop(self, cam, *inputs):
        (inputs, dim, indices) = inputs
        z, grad_func = jax.vjp(lambda i : self.__call__(i, dim, indices), inputs)
        s = safe_divide(cam, z)
        c = grad_func(s)
        outputs = inputs * (c[0])
        return outputs
