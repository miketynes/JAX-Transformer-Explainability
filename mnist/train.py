# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A minimal MNIST classifier example."""

from typing import Iterator, NamedTuple

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

import model

NUM_CLASSES = 10  # MNIST has 10 classes (hand-written digits).


class Batch(NamedTuple):
  image: np.ndarray  # [B, H, W, 1]
  label: np.ndarray  # [B]


class TrainingState(NamedTuple):
  params: hk.Params
  avg_params: hk.Params
  opt_state: optax.OptState

## TODO: Use custom modules
def net_fn(images: jnp.ndarray) -> jnp.ndarray:
  """Standard LeNet-300-100 MLP network."""
  x = images.astype(jnp.float32) / 255.
  flatten = hk.Flatten()
  logits = flatten(x)
  l1 = model.Linear(300)
  logits = jax.nn.relu(l1(logits))
  l2 = model.Linear(100)
  logits = jax.nn.relu(l2(logits))
  output = model.Linear(NUM_CLASSES)
  return output(logits)

def relevence_propogation(images: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
  """Standard LeNet-300-100 MLP network."""
  x = images.astype(jnp.float32) / 255.
  flatten = hk.Flatten()
  logits = flatten(x)

  l1 = model.Linear(300)
  l2_in = jax.nn.relu(l1(logits))

  l2 = model.Linear(100)
  l3_in = jax.nn.relu(l2(l2_in))

  l3 = model.Linear(NUM_CLASSES)
  output = l3(l3_in)

  r = jax.nn.one_hot(c, NUM_CLASSES)
  r = l3.rel_prop(r, l3_in)
  r = l2.rel_prop(r, l2_in)
  r = l1.rel_prop(r, logits)

  return jnp.reshape(r, images.shape)


def load_dataset(
    split: str,
    *,
    shuffle: bool,
    batch_size: int,
) -> Iterator[Batch]:
  """Loads the MNIST dataset."""
  ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
  if shuffle:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  ds = ds.map(lambda x: Batch(**x))
  return iter(tfds.as_numpy(ds))


def main(_):
  # First, make the network and optimiser.
  network = hk.without_apply_rng(hk.transform(net_fn))
  relevence = hk.without_apply_rng(hk.transform(relevence_propogation))
  optimiser = optax.adam(1e-3)

  def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
    """Cross-entropy classification loss, regularised by L2 weight decay."""
    batch_size, *_ = batch.image.shape
    logits = network.apply(params, batch.image)
    labels = jax.nn.one_hot(batch.label, NUM_CLASSES)

    l2_regulariser = 0.5 * sum(
        jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))

    return -log_likelihood / batch_size + 1e-4 * l2_regulariser

  @jax.jit
  def evaluate(params: hk.Params, batch: Batch) -> jnp.ndarray:
    """Evaluation metric (classification accuracy)."""
    logits = network.apply(params, batch.image)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == batch.label)

  @jax.jit
  def update(state: TrainingState, batch: Batch) -> TrainingState:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(state.params, batch)
    updates, opt_state = optimiser.update(grads, state.opt_state)
    params = optax.apply_updates(state.params, updates)
    # Compute avg_params, the exponential moving average of the "live" params.
    # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
    avg_params = optax.incremental_update(
        params, state.avg_params, step_size=0.001)
    return TrainingState(params, avg_params, opt_state)

  # Make datasets.
  train_dataset = load_dataset("train", shuffle=True, batch_size=1_000)
  eval_datasets = {
      split: load_dataset(split, shuffle=False, batch_size=10_000)
      for split in ("train", "test")
  }

  # Initialise network and optimiser; note we draw an input to get shapes.
  initial_params = network.init(
      jax.random.PRNGKey(seed=0), next(train_dataset).image)
  initial_opt_state = optimiser.init(initial_params)
  state = TrainingState(initial_params, initial_params, initial_opt_state)

  # Initialize Relevence Propogation (These params will not be used, just replaced with the trained params)
  _ = relevence.init(
      jax.random.PRNGKey(seed=0), next(train_dataset).image, next(train_dataset).label)

  # Training & evaluation loop.
  for step in range(1001):
    if step % 100 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      # Note that each evaluation is only on a (large) batch.
      for split, dataset in eval_datasets.items():
        accuracy = np.array(evaluate(state.avg_params, next(dataset))).item()
        print({"step": step, "split": split, "accuracy": f"{accuracy:.3f}"})

    # Do SGD on a batch of training examples.
    state = update(state, next(train_dataset))

  test_batch = next(eval_datasets["test"])
  logits = network.apply(state.avg_params, test_batch.image)
  predictions = jnp.argmax(logits, axis=-1)
  errors = (predictions != test_batch.label)
  idx = errors.argmax()

  test_images = test_batch.image[idx:idx+1]
  test_labels = test_batch.label[idx:idx+1]
  print(f"For image {idx}, predicted {predictions[idx]} correct label {test_labels}")  
  rs = []
  for i in range(NUM_CLASSES):
    rs.append(relevence.apply(state.avg_params, test_images, jnp.asarray([i])))

  with open("images.npy", "wb") as fp:
    jnp.save(fp, test_images)
  with open("labels.npy", "wb") as fp:
    jnp.save(fp, test_labels)
  with open("relevence_scores.npy", "wb") as fp:
    jnp.save(fp, jnp.asarray(rs))

if __name__ == "__main__":
  app.run(main)