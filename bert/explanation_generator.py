import argparse
import jax
import jax.numpy as jnp
import numpy as np
import glob

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = jnp.eye(num_tokens)[jnp.newaxis]
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(axis=-1, keepdims=True)
                          for i in range(len(all_layer_matrices))]
    
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = jnp.einsum('...ij,...jk', matrices_aug[i], joint_attention)
    return joint_attention

class Generator:
    def __init__(self, model):
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def generate_LRP(self, input_ids, attention_mask,
                     index=None, start_layer=11):

        def loss_fn(input_ids, attention_mask, perturbs, index=None):
            output = self.model(input_ids, attention_mask, perturbs=perturbs)[0]
            if index == None:
                index = jnp.argmax(output, axis=-1)

            one_hot = np.zeros((1, output.shape[-1]), dtype=jnp.float32)
            one_hot[0, index] = 1
            one_hot_vector = jnp.array(one_hot)
            loss = jnp.sum(one_hot * output)
            return loss

        intermediate_grads = jax.grad(loss_fn, argnums=2)(input_ids, attention_mask, self.model.perturbs)
        
        output = self.model(input_ids, attention_mask)[0]
        if index == None:
            index = jnp.argmax(output, axis=-1)

        one_hot = np.zeros((1, output.shape[-1]), dtype=jnp.float32)
        one_hot[0, index] = 1
        one_hot_vector = jnp.array(one_hot)
        input_cam, attn_cams = self.model.relprop(one_hot_vector, input_ids, attention_mask)

        cams = []
        for blk, cam in zip(intermediate_grads["bert"]["encoder"]["layer"].values(), reversed(attn_cams)):
            grad = blk["attention"]["self"]["attn_weights"]
            cam = jnp.reshape(cam[0], (-1, cam.shape[-1], cam.shape[-1]))
            grad = jnp.reshape(grad[0], (-1, grad.shape[-1], grad.shape[-1]))
            cam = grad * cam
            cam = jnp.clip(cam, a_min=0).mean(axis=0)
            cams.append(cam[jnp.newaxis])

        rollout = compute_rollout_attention(cams, start_layer=start_layer)
        rollout = rollout.at[:, 0, 0].set(rollout[:, 0].min())
        return rollout[:, 0]