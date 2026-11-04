"""Reference instrumentation for Transformers 5.22.1 Qwen3.5.

Adapted from Hugging Face Transformers modeling_qwen3_5.py, copyright the
Hugging Face Inc. team, Apache License 2.0. See
https://github.com/huggingface/transformers/blob/446a570e62b04a89ea001604ada7366baac2ea67/src/transformers/models/qwen3_5/modeling_qwen3_5.py .
The accompanying LICENSE.transformers retains the upstream Apache license.
This module is deliberately version-specific; it is not a replacement model.

Recorder protocol: ``record(name, tensor, inputs=(), token=None)``. ``inputs``
names earlier records or module parameters; ``token`` marks per-token
recurrent records. The recorder must select/copy bounded slices immediately;
tensors are transient and may be mutated by later operations.

The gated-delta adapter has two algorithms:

* ``upstream`` (default) dispatches exactly like the pinned implementation:
  ``torch_chunk_gated_delta_rule`` for prefill and
  ``torch_recurrent_gated_delta_rule`` for one-token cached decoding. The
  explicit token-by-token recurrence still runs to expose states, but only as
  an observation; the layer output is the upstream kernel output.
* ``recurrent`` uses the explicit recurrence output instead. This is a
  different (mathematically equivalent) kernel and is used to measure
  chunk-versus-recurrent numerical error, not to claim exact parity.
"""
from contextlib import contextmanager
from types import MethodType
import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream

ALGORITHMS = ("upstream", "recurrent")


def _check():
    if transformers.__version__ != "5.22.1":
        raise RuntimeError(f"Qwen3.5 adapter requires transformers 5.22.1; got {transformers.__version__}")


def attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, *, record, **kwargs):
    shape = hidden_states.shape[:-1]
    hshape = (*shape, -1, self.head_dim)
    record("mixer_input", hidden_states, ("input_layernorm",))
    raw = self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2)
    q, gate = torch.chunk(raw, 2, dim=-1)
    record("query_projection", q, ("mixer_input", "q_proj.weight"))
    record("output_gate_logits", gate, ("mixer_input", "q_proj.weight"))
    k_raw = self.k_proj(hidden_states).view(hshape)
    v_raw = self.v_proj(hidden_states).view(hshape)
    record("key_projection", k_raw, ("mixer_input", "k_proj.weight"))
    record("value_projection", v_raw, ("mixer_input", "v_proj.weight"))
    q = self.q_norm(q.reshape(hshape)).transpose(1, 2)
    k = self.k_norm(k_raw).transpose(1, 2)
    v = v_raw.transpose(1, 2)
    record("query_normalized", q, ("query_projection", "q_norm.weight"))
    record("key_normalized", k, ("key_projection", "k_norm.weight"))
    record("value", v, ("value_projection",))
    record("rotary_cos", position_embeddings[0], ("position_ids",))
    record("rotary_sin", position_embeddings[1], ("position_ids",))
    q, k = upstream.apply_rotary_pos_emb(q, k, *position_embeddings)
    record("query_rotary", q, ("query_normalized", "rotary_cos", "rotary_sin"))
    record("key_rotary", k, ("key_normalized", "rotary_cos", "rotary_sin"))
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)
    k = upstream.repeat_kv(k, self.num_key_value_groups)
    v = upstream.repeat_kv(v, self.num_key_value_groups)
    record("key_grouped", k, ("key_rotary", "kv_cache"))
    record("value_grouped", v, ("value", "kv_cache"))
    scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
    record("scores", scores, ("query_rotary", "key_grouped"))
    if attention_mask is not None:
        record("mask", attention_mask, ("attention_mask",))
        scores = scores + attention_mask
    record("masked_scores", scores, ("scores", "mask"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    probs = F.dropout(probs, p=0.0 if not self.training else self.attention_dropout, training=self.training)
    record("probabilities", probs, ("masked_scores",))
    out = torch.matmul(probs, v).transpose(1, 2).contiguous()
    record("value_aggregation", out, ("probabilities", "value_grouped"))
    out = out.reshape(*shape, -1).contiguous()
    gate = gate.reshape(*shape, -1)
    record("output_gate", torch.sigmoid(gate), ("output_gate_logits",))
    out = out * torch.sigmoid(gate)
    record("gated_output", out, ("value_aggregation", "output_gate"))
    out = self.o_proj(out)
    record("projection", out, ("gated_output", "o_proj.weight"))
    return out, probs


def explicit_recurrence(q, k, v, g, beta, state, record=None):
    """Token-by-token gated delta rule on [batch, head, token, feature] FP32 tensors.

    Mirrors torch_recurrent_gated_delta_rule after its L2 normalization and query
    scaling; records the decayed state, readout memory, delta and updated state.
    """
    out = torch.zeros_like(v)
    for i in range(q.shape[2]):
        if record is not None:
            record("state_before", state, ("recurrent_state",), token=i)
        state = state * g[:, :, i].exp()[..., None, None]
        memory = (state * k[:, :, i].unsqueeze(-1)).sum(dim=-2)
        delta = (v[:, :, i] - memory) * beta[:, :, i].unsqueeze(-1)
        state = state + k[:, :, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q[:, :, i].unsqueeze(-1)).sum(dim=-2)
        if record is not None:
            record("state_memory_readout", memory, ("state_before", "log_decay", "key_normalized"), token=i)
            record("state_delta", delta, ("value", "state_memory_readout", "beta"), token=i)
            record("state_after", state, ("state_before", "key_normalized", "state_delta"), token=i)
    return out, state


def delta_forward(self, hidden_states, cache_params=None, attention_mask=None, *, record, algorithm="upstream",
                  explain=True, **kwargs):
    record("mixer_input", hidden_states, ("input_layernorm",))
    hidden_states = upstream.apply_mask_to_padding_states(hidden_states, attention_mask)
    record("padding_masked_input", hidden_states, ("mixer_input", "attention_mask"))
    batch, length, _ = hidden_states.shape
    conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0) if cache_params is not None else None
    state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0) if cache_params is not None else None
    mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
    record("pre_convolution_qkv", mixed, ("padding_masked_input", "in_proj_qkv.weight"))
    z = self.in_proj_z(hidden_states).reshape(batch, length, -1, self.head_v_dim)
    b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    record("output_gate_logits", z, ("padding_masked_input", "in_proj_z.weight"))
    record("beta_logits", b, ("padding_masked_input", "in_proj_b.weight"))
    record("decay_logits", a, ("padding_masked_input", "in_proj_a.weight"))
    if conv_state is not None and length == 1 and not cache_params.is_recording_past(self.layer_idx):
        record("convolution_history", conv_state, ("conv_cache",))
        mixed = upstream.causal_conv1d_update(mixed, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
    else:
        if cache_params is not None:
            mixed = cache_params.update_conv_state(mixed, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
        mixed = upstream.causal_conv1d_fn(mixed, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation, **kwargs)
        if cache_params is not None:
            mixed = mixed[:, :, -length:]
    mixed = mixed.transpose(1, 2)
    record("post_convolution_qkv", mixed, ("pre_convolution_qkv", "conv1d.weight", "conv_cache"))
    q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(batch, length, -1, self.head_k_dim)
    k = k.reshape(batch, length, -1, self.head_k_dim)
    v = v.reshape(batch, length, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    record("beta", beta, ("beta_logits",)); record("log_decay", g, ("decay_logits", "A_log", "dt_bias"))
    record("value", v, ("post_convolution_qkv",))
    if self.num_v_heads // self.num_k_heads > 1:
        q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    initial_dtype = q.dtype
    if algorithm == "upstream":
        rule = upstream.torch_recurrent_gated_delta_rule if state is not None and length == 1 else upstream.torch_chunk_gated_delta_rule
        kernel_out, kernel_state = rule(q, k, v, g=g, beta=beta, initial_state=state,
                                        output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                                        cu_seqlens=kwargs.pop("cu_seq_lens_q", None), **kwargs)
        record("kernel_readout", kernel_out, ("query_normalized_scaled", "key_normalized", "value", "beta", "log_decay"))
    if algorithm == "recurrent" or explain:
        fq, fk, fv, fbeta, fg = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format) for x in (q, k, v, beta, g)]
        fq, fk = upstream.l2norm(fq, dim=-1, eps=1e-6), upstream.l2norm(fk, dim=-1, eps=1e-6)
        fq = fq / (fq.shape[-1] ** 0.5)
        record("query_normalized_scaled", fq, ("post_convolution_qkv",))
        record("key_normalized", fk, ("post_convolution_qkv",))
        initial = torch.zeros((batch, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=fv.dtype, device=fv.device) \
            if state is None else state.to(fv)
        loop_out, loop_state = explicit_recurrence(fq, fk, fv, fg, fbeta, initial, record)
        record("readout", loop_out, ("state_after", "query_normalized_scaled"))
    if algorithm == "recurrent":
        out, final_state = loop_out.transpose(1, 2).contiguous().to(initial_dtype), loop_state
    elif algorithm == "upstream":
        out, final_state = kernel_out, kernel_state
    else:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    if cache_params is not None:
        cache_params.update_recurrent_state(final_state, self.layer_idx)
    out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    out = out.reshape(batch, length, -1)
    record("gated_output", out, ("kernel_readout" if algorithm == "upstream" else "readout", "output_gate_logits", "norm.weight"))
    out = self.out_proj(out)
    record("projection", out, ("gated_output", "out_proj.weight"))
    return out


@contextmanager
def instrument(module, kind, recorder, algorithm="upstream", explain=True):
    """Temporarily replace a single mixer's forward, restoring it on exceptions."""
    _check()
    if kind not in ("attention", "delta"):
        raise ValueError("kind must be attention or delta")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    if "forward" in module.__dict__:
        raise RuntimeError("Mixer is already instrumented; nested instrumentation is not supported")
    def forward(self, *args, **kwargs):
        if kind == "attention":
            return attention_forward(self, *args, record=recorder, **kwargs)
        return delta_forward(self, *args, record=recorder, algorithm=algorithm, explain=explain, **kwargs)
    module.forward = MethodType(forward, module)
    try:
        yield
    finally:
        del module.forward  # fall back to the class method; leaves no instance attribute behind
