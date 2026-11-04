"""Reference instrumentation for Transformers 5.22.1 Qwen3.5.

Adapted from Hugging Face Transformers modeling_qwen3_5.py, copyright the
Hugging Face Inc. team, Apache License 2.0. See
https://github.com/huggingface/transformers/blob/446a570e62b04a89ea001604ada7366baac2ea67/src/transformers/models/qwen3_5/modeling_qwen3_5.py .
The accompanying LICENSE.transformers retains the upstream Apache license.
This module is deliberately version-specific; it is not a replacement model.
The recorder must select/copy bounded slices immediately; tensors are transient.
"""
from contextlib import contextmanager
from types import MethodType
import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream


def _check():
    if transformers.__version__ != "5.22.1":
        raise RuntimeError(f"Qwen3.5 adapter requires transformers 5.22.1; got {transformers.__version__}")


def attention_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, *, record, **kwargs):
    shape = hidden_states.shape[:-1]
    hshape = (*shape, -1, self.head_dim)
    raw = self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2)
    q, gate = torch.chunk(raw, 2, dim=-1)
    record("query_projection", q); record("output_gate_logits", gate)
    k_raw = self.k_proj(hidden_states).view(hshape)
    v_raw = self.v_proj(hidden_states).view(hshape)
    record("key_projection", k_raw); record("value_projection", v_raw)
    q = self.q_norm(q.view(hshape)).transpose(1, 2)
    k = self.k_norm(k_raw).transpose(1, 2)
    v = v_raw.transpose(1, 2)
    record("query_normalized", q); record("key_normalized", k); record("value", v)
    record("rotary_cos", position_embeddings[0]); record("rotary_sin", position_embeddings[1])
    q, k = upstream.apply_rotary_pos_emb(q, k, *position_embeddings)
    record("query_rotary", q); record("key_rotary", k)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)
    k = upstream.repeat_kv(k, self.num_key_value_groups)
    v = upstream.repeat_kv(v, self.num_key_value_groups)
    record("key_grouped", k); record("value_grouped", v)
    scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
    record("scores", scores)
    if attention_mask is not None:
        scores = scores + attention_mask
        record("mask", attention_mask)
    record("masked_scores", scores)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    probs = F.dropout(probs, p=0.0 if not self.training else self.attention_dropout, training=self.training)
    record("probabilities", probs)
    out = torch.matmul(probs, v).transpose(1, 2).contiguous()
    record("value_aggregation", out)
    out = out.reshape(*shape, -1).contiguous()
    out = out * torch.sigmoid(gate.reshape(*shape, -1))
    record("gated_output", out)
    out = self.o_proj(out)
    record("projection", out)
    return out, probs


def delta_forward(self, hidden_states, cache_params=None, attention_mask=None, *, record, **kwargs):
    hidden_states = upstream.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    conv_state = cache_params.get_conv_state(self.layer_idx, state_idx=0) if cache_params is not None else None
    state = cache_params.get_recurrent_state(self.layer_idx, state_idx=0) if cache_params is not None else None
    mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
    record("pre_convolution_qkv", mixed)
    z = self.in_proj_z(hidden_states).reshape(batch, length, -1, self.head_v_dim)
    b, a = self.in_proj_b(hidden_states), self.in_proj_a(hidden_states)
    if conv_state is not None and length == 1 and not cache_params.is_recording_past(self.layer_idx):
        mixed = upstream.causal_conv1d_update(mixed, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
    else:
        if cache_params is not None:
            mixed = cache_params.update_conv_state(mixed, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
        mixed = upstream.causal_conv1d_fn(mixed, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation, **kwargs)
        if cache_params is not None:
            mixed = mixed[:, :, -length:]
    mixed = mixed.transpose(1, 2)
    record("post_convolution_qkv", mixed)
    q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(batch, length, -1, self.head_k_dim)
    k = k.reshape(batch, length, -1, self.head_k_dim)
    v = v.reshape(batch, length, -1, self.head_v_dim)
    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    record("beta", beta); record("log_decay", g); record("value", v); record("gate", z)
    if self.num_v_heads // self.num_k_heads > 1:
        q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    initial_dtype = q.dtype
    q, k, v, beta, g = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format) for x in (q, k, v, beta, g)]
    q, k = upstream.l2norm(q, dim=-1, eps=1e-6), upstream.l2norm(k, dim=-1, eps=1e-6)
    q = q / (q.shape[-1] ** 0.5)
    record("query_normalized_scaled", q); record("key_normalized", k)
    if state is None:
        state = torch.zeros((batch, self.num_v_heads, self.head_k_dim, self.head_v_dim), dtype=v.dtype, device=v.device)
    else:
        state = state.to(v)
    out = torch.zeros_like(v)
    for i in range(length):
        state = state * g[:, :, i].exp()[..., None, None]
        memory = (state * k[:, :, i].unsqueeze(-1)).sum(dim=-2)
        delta = (v[:, :, i] - memory) * beta[:, :, i].unsqueeze(-1)
        state = state + k[:, :, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q[:, :, i].unsqueeze(-1)).sum(dim=-2)
        record(f"state_after_token_{i}", state); record(f"delta_token_{i}", delta)
    record("readout", out)
    if cache_params is not None:
        cache_params.update_recurrent_state(state, self.layer_idx)
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    out = self.norm(out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    out = out.reshape(batch, length, -1)
    record("gated_output", out)
    out = self.out_proj(out)
    record("projection", out)
    return out


@contextmanager
def instrument(module, kind, recorder):
    """Temporarily replace a single mixer's forward, restoring it on exceptions."""
    _check()
    if kind not in ("attention", "delta"):
        raise ValueError("kind must be attention or delta")
    original = module.forward
    implementation = attention_forward if kind == "attention" else delta_forward
    def forward(self, *args, **kwargs):
        return implementation(self, *args, record=recorder, **kwargs)
    module.forward = MethodType(forward, module)
    try:
        yield
    finally:
        module.forward = original
