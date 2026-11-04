"""Exact checkpoint, loaded storage, observed text-path, and cache accounting."""
from __future__ import annotations
from collections import defaultdict
import gc
import inspect
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open
from .runtime import load_model, write_json


def storage_key(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def cache_tensors(cache):
    """Walk cache objects without assuming a particular Transformers cache layout."""
    found, visited = [], set()
    def visit(obj, name):
        if torch.is_tensor(obj):
            found.append((name, obj)); return
        if id(obj) in visited: return
        visited.add(id(obj))
        if isinstance(obj, dict):
            for k, v in obj.items(): visit(v, f"{name}.{k}")
        elif isinstance(obj, (list, tuple)):
            for k, v in enumerate(obj): visit(v, f"{name}.{k}")
        elif hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
            for k, v in vars(obj).items():
                if not k.startswith("__"): visit(v, f"{name}.{k}")
    visit(cache, "cache")
    return found


def build_inventory(config=None, output="runs/inventory", cache_lengths=(8, 32, 64)):
    rt = load_model(config)
    model = rt.model
    modules = dict(model.named_modules())
    records = []
    storage_refs = defaultdict(list)
    for name, param in model.named_parameters(remove_duplicate=False):
        key = storage_key(param)
        storage_refs[key].append(name)
        records.append({"name": name, "shape": list(param.shape), "numel": param.numel(),
                        "bytes": param.numel() * param.element_size(), "storage_bytes": param.untyped_storage().nbytes(),
                        "dtype": str(param.dtype), "device": str(param.device), "trainable": param.requires_grad})
    checkpoint = []
    for file in sorted(rt.model_dir.glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_slice(name)
                shape = tensor.get_shape()
                checkpoint.append({"name": name, "shape": shape, "numel": math.prod(shape),
                                   "dtype": tensor.get_dtype(), "file": file.name,
                                   "bytes": math.prod(shape) * {"BF16":2,"F16":2,"F32":4,"F64":8,"I64":8,"I32":4,"I8":1,"U8":1,"BOOL":1}.get(tensor.get_dtype(), 0)})
    executed = set()
    observed = {}
    handles = []
    def summarize(value):
        if torch.is_tensor(value): return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
        if isinstance(value, (list, tuple)): return [summarize(x) for x in value[:3]]
        return type(value).__name__
    def hook(name):
        def record(mod, args, result):
            executed.add(name)
            observed[name] = {"input": summarize(args), "output": summarize(result)}
        return record
    for name, module in modules.items(): handles.append(module.register_forward_hook(hook(name)))
    inputs = rt.tokenizer("The capital of France is", return_tensors="pt").to(rt.device)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for h in handles: h.remove()
    active_keys = set()
    for name in executed:
        for param in modules[name].parameters(recurse=False): active_keys.add(storage_key(param))
        # Qwen3.5 invokes convolution functionally; its conv1d module hook does
        # not fire, although its weights participate in every linear mixer.
        if name.endswith(".linear_attn"):
            for param in modules[name].conv1d.parameters(): active_keys.add(storage_key(param))
    by_component = defaultdict(lambda: {"parameters": 0, "bytes": 0})
    unique_parameters = {}
    for name, param in model.named_parameters():
        key = storage_key(param)
        if key in unique_parameters: continue
        unique_parameters[key] = param.numel()
        component = "vision" if ".visual." in name or name.startswith("visual.") else "embedding" if "embed_tokens" in name else "text_mixer" if any(x in name for x in ["linear_attn", "self_attn"]) else "text_mlp" if ".mlp." in name else "other"
        by_component[component]["parameters"] += param.numel()
        by_component[component]["bytes"] += param.untyped_storage().nbytes()
    caches = []
    for length in cache_lengths:
        ids = rt.tokenizer("A", add_special_tokens=False, return_tensors="pt").input_ids[:, :1].repeat(1, length).to(rt.device)
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
        tensors = cache_tensors(result.past_key_values)
        unique = {}
        entries = []
        for name, tensor in tensors:
            unique[storage_key(tensor)] = tensor.untyped_storage().nbytes()
            entries.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "bytes": tensor.numel() * tensor.element_size(), "storage_bytes": tensor.untyped_storage().nbytes()})
        text_config = model.config.text_config
        full_count = text_config.layer_types.count("full_attention")
        linear_count = text_config.layer_types.count("linear_attention")
        conv_example = next(x for x in entries if "conv_states" in x["name"])
        recurrent_example = next(x for x in entries if "recurrent_states" in x["name"])
        conv_dtype_bytes = conv_example["bytes"] // math.prod(conv_example["shape"])
        state_dtype_bytes = recurrent_example["bytes"] // math.prod(recurrent_example["shape"])
        model_dtype_bytes = next(model.parameters()).element_size()
        kv_calculated = full_count * 2 * length * text_config.num_key_value_heads * text_config.head_dim * model_dtype_bytes
        state_calculated = linear_count * text_config.linear_num_value_heads * text_config.linear_key_head_dim * text_config.linear_value_head_dim * state_dtype_bytes
        conv_calculated = linear_count * conv_example["shape"][-2] * conv_example["shape"][-1] * conv_dtype_bytes
        caches.append({"sequence_length": length, "batch_size": 1, "measured_storage_bytes": sum(unique.values()),
                       "tensor_bytes": sum(x["bytes"] for x in entries), "tensors": entries,
                       "calculated": {"full_kv_bytes": kv_calculated, "delta_recurrent_bytes": state_calculated,
                                      "delta_convolution_bytes": conv_calculated,
                                      "total_logical_bytes": kv_calculated + state_calculated + conv_calculated,
                                      "storage_minus_logical_bytes": sum(unique.values()) - sum(x["bytes"] for x in entries)},
                       "formula": {"full_per_layer": "2 * B * T * num_key_value_heads * head_dim * dtype_bytes",
                                   "delta_recurrent_per_layer": "B * value_heads * key_head_dim * value_head_dim * state_dtype_bytes",
                                   "delta_conv_per_layer": "B * observed_stored_width * observed_history_length * conv_dtype_bytes",
                                   "observed_conv_shape": conv_example["shape"], "observed_state_shape": recurrent_example["shape"],
                                   "state_dtype_bytes": state_dtype_bytes, "conv_dtype_bytes": conv_dtype_bytes,
                                   "full_layer_count": full_count, "linear_layer_count": linear_count,
                                   "allocator_note": "Tensor storage measured here excludes CUDA caching allocator reservation and Python object overhead."}})
        del result, tensors, tensor
        gc.collect()
    layers = []
    for i, layer in enumerate(model.model.language_model.layers):
        mixer = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
        path = f"model.language_model.layers.{i}"
        layers.append({"index": i, "path": path, "type": layer.block_type,
                       "mixer_class": type(mixer).__name__, "parameters": sum(p.numel() for p in layer.parameters()),
                       "weight_shapes": {n: list(p.shape) for n, p in layer.named_parameters()},
                       "observed": {n: obs for n, obs in observed.items() if n.startswith(path)}})
    module_records = []
    for name, module in modules.items():
        module_records.append({"path": name, "class": type(module).__name__, "direct_parameters": sum(p.numel() for p in module.parameters(recurse=False)),
                               "executed_text": name in executed, "observed": observed.get(name)})
    loaded_names = {x["name"] for x in records}
    checkpoint_names = {x["name"] for x in checkpoint}
    checkpoint_only = sorted(checkpoint_names - loaded_names)
    omitted_parameters = sum(x["numel"] for x in checkpoint if x["name"] in checkpoint_only)
    implementation = Path(rt.provenance["transformers_source_path"]).read_text().splitlines()
    source_line = lambda needle: next(i for i, text in enumerate(implementation, 1) if needle in text)
    aliases = [names for names in storage_refs.values() if len(names) > 1]
    report = {"kind": "inventory", "config": model.config.to_dict(), "layers": layers,
              "modules": module_records, "parameters": records, "checkpoint": checkpoint,
              "accounting": {"checkpoint_tensor_parameters": sum(x["numel"] for x in checkpoint),
                             "checkpoint_tensor_bytes": sum(x["bytes"] for x in checkpoint),
                             "checkpoint_tensor_count": len(checkpoint), "loaded_unique_parameters": sum(unique_parameters.values()),
                             "loaded_logical_parameters": sum(x["numel"] for x in records),
                             "loaded_unique_storage_bytes": sum(k[2] for k in storage_refs),
                             "active_text_unique_parameters": sum(unique_parameters[k] for k in active_keys),
                             "inactive_unique_parameters": sum(v for k, v in unique_parameters.items() if k not in active_keys),
                             "active_text_unique_storage_bytes": sum(k[2] for k in active_keys),
                             "tied_storage_references": aliases, "by_component": dict(by_component),
                             "checkpoint_only_keys": checkpoint_only,
                             "loaded_only_keys": sorted(loaded_names - checkpoint_names),
                             "loading_info": rt.loading_info,
                             "active_definition": "Unique direct parameter storages on modules observed executing a text-only forward, plus convolution weights accessed functionally by executed linear mixers; tied head/embed storage counted once.",
                             "functional_access_evidence": {"operation": "causal_conv1d_fn/causal_conv1d_update receives self.conv1d.weight",
                                                            "source_line": source_line("self.conv1d.weight.squeeze(1)"),
                                                            "module_pattern": "model.language_model.layers.*.linear_attn.conv1d"},
                             "reconciliation": {"checkpoint_only_parameters": omitted_parameters,
                                                "loaded_equals_checkpoint_minus_omitted": sum(unique_parameters.values()) == sum(x["numel"] for x in checkpoint) - omitted_parameters,
                                                "equation": f"{sum(x['numel'] for x in checkpoint)} checkpoint - {omitted_parameters} MTP = {sum(unique_parameters.values())} loaded; loaded - {sum(v for k,v in unique_parameters.items() if k not in active_keys)} inactive vision = {sum(unique_parameters[k] for k in active_keys)} active text",
                                                "mtp_loading_policy": "Upstream deliberately ignores ^mtp.* unexpected keys; empty loader unexpected_keys does not mean MTP was loaded.",
                                                "mtp_ignore_source_line": source_line("_keys_to_ignore_on_load_unexpected"),
                                                "tied_logical_reference_note": "lm_head.weight is a loaded-only logical alias of embed_tokens.weight and contributes no extra storage."}},
              "cache_measurements": caches, "provenance": rt.provenance}
    write_json(Path(output) / "inventory.json", report)
    return report
