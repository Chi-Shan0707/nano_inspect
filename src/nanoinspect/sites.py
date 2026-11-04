"""Semantic decoder sites. Paths are resolved from the loaded model, not assumed.

``post_mixer_residual`` is the input to post_attention_layernorm, i.e. the
sum of the mixer output and the incoming residual in Qwen3.5's decoder block.
Capturing that input is exact. An intervention there must also update the
residual path, which upstream copies before the norm; see
``interventions.Intervention``.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    name: str
    layer: int
    path: str
    module: object
    boundary: str  # input or output
    axes: tuple = ("batch", "token", "hidden")


def decoder_layers(model):
    """Find the unique text decoder ModuleList by its decoder-block members."""
    candidates = []
    for path, module in model.named_modules():
        if module.__class__.__name__ == "ModuleList" and len(module):
            first = module[0]
            if all(hasattr(first, x) for x in ("input_layernorm", "post_attention_layernorm", "mlp")):
                candidates.append((path, module))
    if len(candidates) != 1:
        raise ValueError(f"Expected one text decoder layer list, found {[p for p, _ in candidates]}")
    return candidates[0]


def resolve_site(model, layer, name):
    path, layers = decoder_layers(model)
    if layer < 0 or layer >= len(layers):
        raise IndexError(f"Layer {layer} outside [0, {len(layers)})")
    block = layers[layer]
    prefix = f"{path}.{layer}"
    if name == "residual_input":
        return Site(name, layer, prefix, block, "input")
    if name == "residual_output":
        return Site(name, layer, prefix, block, "output")
    if name == "post_mixer_residual":
        return Site(name, layer, prefix + ".post_attention_layernorm", block.post_attention_layernorm, "input")
    if name == "mlp_output":
        return Site(name, layer, prefix + ".mlp", block.mlp, "output")
    if name == "mixer_output":
        block_type = getattr(block, "block_type", getattr(block, "layer_type", None))
        mixer_name = "self_attn" if block_type == "full_attention" else "linear_attn"
        if not hasattr(block, mixer_name):
            raise ValueError(f"Unsupported mixer on {prefix}: {block_type}")
        return Site(name, layer, prefix + "." + mixer_name, getattr(block, mixer_name), "output")
    raise ValueError(f"Unknown site {name!r}")


SITE_NAMES = ("residual_input", "mixer_output", "post_mixer_residual", "mlp_output", "residual_output")
