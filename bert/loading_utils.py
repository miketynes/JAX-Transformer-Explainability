import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict


def load_params_from_hf(pt_params, hidden_size, num_attention_heads):
  """from flax_bert repo, here: https://github.com/nikitakit/flax_bert/blob/57c2b89c0e49b22ac9ae14800f3cbb1782b8f979/import_weights.py#L109"""
  jax_params = {}
  # mapping between HuggingFace PyTorch BERT and JAX model
  pt_key_to_jax_key = [
    # Output heads
    ('cls.seq_relationship', 'classification'),
    ('cls.predictions.transform.LayerNorm', 'predictions_transform_layernorm'),
    ('cls.predictions.transform.dense', 'predictions_transform_dense'),
    ('cls.predictions.bias', 'predictions_output.bias'),
    ('cls.predictions.decoder.weight', 'UNUSED'),
    ('cls.predictions.decoder.bias', 'UNUSED'),
    # Embeddings
    ('embeddings.position_ids', 'UNUSED'),
    ('embeddings.word_embeddings.weight', 'word_embeddings.embedding'),
    ('embeddings.token_type_embeddings.weight', 'type_embeddings.embedding'),
    ('embeddings.position_embeddings.weight', 'position_embeddings.embedding'),
    ('embeddings.LayerNorm', 'embeddings_layer_norm'),
    # Pooler
    ('pooler.dense.', 'pooler.'),
    # Layers
    ('bert.encoder.layer.', 'bert.encoder_layer_'),
    # ('bert/encoder/layer_', 'bert/encoder_layer_'),
    ('attention.self', 'self_attention.attn'),
    ('attention.output.dense', 'self_attention.attn.output'),
    ('attention.output.LayerNorm', 'self_attention_layer_norm'),
    ('output.LayerNorm', 'output_layer_norm'),
    ('intermediate.dense', 'feed_forward.intermediate'),
    ('output.dense', 'feed_forward.output'),
    # Parameter names
    ('weight', 'kernel'),
    ('beta', 'bias'),
    ('gamma', 'scale'),
    ('layer_norm.kernel', 'layer_norm.scale'),
    ('layernorm.kernel', 'layernorm.scale'),
  ]
  pt_keys_to_transpose = (
    "dense.weight",
    "attention.self.query",
    "attention.self.key",
    "attention.self.value"
  )
  for pt_key, val in pt_params.items():
    jax_key = pt_key
    for pt_name, jax_name in pt_key_to_jax_key:
      jax_key = jax_key.replace(pt_name, jax_name)

    if 'UNUSED' in jax_key:
      continue

    if any([x in pt_key for x in pt_keys_to_transpose]):
      val = val.T
    val = np.asarray(val)

    # Reshape kernels if necessary
    reshape_params = ['key', 'query', 'value']
    for key in reshape_params:
      if f'self_attention.attn.{key}.kernel' in jax_key:
        val = np.swapaxes(
          val.reshape((hidden_size, num_attention_heads, -1)), 0, 1)
      elif f'self_attention.attn.{key}.bias' in jax_key:
        val = val.reshape((num_attention_heads, -1))
    if 'self_attention.attn.output.kernel' in jax_key:
      val = val.reshape((num_attention_heads, -1, hidden_size))
    elif 'self_attention.attn.output.bias' in jax_key:
      # The multihead attention implementation we use creates a bias vector for
      # each head, even though this is highly redundant.
      val = np.stack(
        [val] + [np.zeros_like(val)] * (num_attention_heads - 1), axis=0)

    jax_params[jax_key] = val

  # jax position embedding kernel has additional dimension
  pos_embedding = jax_params[
    'bert.position_embeddings.embedding']
  jax_params[
    'bert.position_embeddings.embedding'] = pos_embedding[
    np.newaxis, ...]

  # this layer doesn't have parameters, but key is required to be present
  jax_params['GatherIndexes_0'] = {}

  # convert flat param dict into nested dict using `/` as delimeter
  outer_dict = {}
  for key, val in jax_params.items():
    tokens = key.split('.')
    inner_dict = outer_dict
    # each token except the very last should add a layer to the nested dict
    for token in tokens[:-1]:
      if token not in inner_dict:
        inner_dict[token] = {}
      inner_dict = inner_dict[token]
    inner_dict[tokens[-1]] = val

  if 'global_step' in outer_dict:
    del outer_dict['global_step']

  return outer_dict

def clean_state_dict(processed_state_dict):
    """cleans the keys of the state dict from load_params_from_hf s.t. they match the names in FlaxBertForSequenceClassification"""
    state_dict_flat = flatten_dict(processed_state_dict)
    clean_state_dict = {}
    for key in state_dict_flat.keys():
      new_key = list(key)
      if 'encoder_layer_' in key[1]:
        new_key = [new_key[0]] + new_key[1].split('_') + new_key[2:]
      if new_key[1] == 'embeddings_layer_norm':
        new_key = [new_key[0]] + ['embeddings', 'LayerNorm'] + [new_key[-1]]
      if new_key[1] in ['position_embeddings',
                        'word_embeddings',
                        'type_embeddings']:
        if new_key[1] == 'type_embeddings':
          new_key[1] = 'token_type_embeddings'
        new_key = [new_key[0]] + ['embeddings'] + new_key[1:]
      if new_key[1] == 'pooler':
        new_key.insert(-1, 'dense')
      if len(key) > 3:
        if new_key[4] == 'feed_forward':
          new_key[4] = 'dense'
        if new_key[4] == 'self_attention':
          new_key[4] = 'attention'
          new_key[5] = 'self'
          if new_key[6] == 'output':
            new_key[5] = 'output'
            new_key[6] = 'dense'
        if new_key[-3:-1] == ['dense', 'intermediate']:
          new_key[-3] = 'intermediate'
          new_key[-2] = 'dense'
        if new_key[-2] == 'output_layer_norm':
          new_key = new_key[:-2] + ['output', 'LayerNorm'] + new_key[-1:]
        if new_key[-2] == 'self_attention_layer_norm':
          new_key = new_key[:-2] + ['attention', 'output', 'LayerNorm'] + new_key[-1:]
      if len(new_key) == 7:
        if '.'.join(new_key[:3]) == 'bert.encoder.layer' and \
                '.'.join(new_key[-3:]) in ['dense.output.bias', 'dense.output.kernel']:
          new_key[-3] = 'output'
          new_key[-2] = 'dense'
      clean_state_dict[tuple(new_key)] = state_dict_flat[key]
    return unflatten_dict(clean_state_dict)

def flax_bert_params_from_pt(pt_params, hidden_size, num_attention_heads):
  params = load_params_from_hf(pt_params, hidden_size, num_attention_heads)
  return clean_state_dict(params)

