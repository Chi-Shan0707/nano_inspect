"""Bounded, exception-safe capture at semantic decoder sites."""
import torch
from .sites import resolve_site

# Every capture record carries these keys (PLAN.md M3).
REQUIRED_METADATA = ("run", "layer", "token_ids", "positions", "phase", "site", "shape", "dtype", "model_revision")


def first_tensor(value):
    return value[0] if isinstance(value, tuple) else value


def replace_first(value, tensor):
    return (tensor, *value[1:]) if isinstance(value, tuple) else tensor


def normalize_positions(positions, length):
    """Resolve explicit (possibly negative) token positions; ``"all"`` selects every position."""
    if isinstance(positions, str):
        if positions != "all":
            raise ValueError(f"Unknown position selector {positions!r}")
        return list(range(length))
    indices = [p if p >= 0 else length + p for p in positions]
    if any(p < 0 or p >= length for p in indices):
        raise IndexError(f"Token position {list(positions)} outside sequence length {length}")
    return indices


def per_row(positions):
    """True when ``positions`` gives one aligned position list per batch row."""
    return not isinstance(positions, str) and len(positions) > 0 and isinstance(positions[0], (list, tuple))


def row_indices(positions, batch, length):
    """``[batch, k]`` index tensor for shared positions or aligned per-row position lists."""
    if not per_row(positions):
        return torch.tensor([normalize_positions(positions, length)] * batch, dtype=torch.long)
    if len(positions) != batch:
        raise ValueError(f"{len(positions)} per-row position lists for batch size {batch}")
    rows = [normalize_positions(p, length) for p in positions]
    if len({len(r) for r in rows}) != 1:
        raise ValueError("Per-row position lists must have equal lengths")
    return torch.tensor(rows, dtype=torch.long)


def select_positions(tensor, positions):
    if tensor.ndim != 3:
        raise ValueError(f"Expected [batch, token, hidden], got {tuple(tensor.shape)}")
    index = row_indices(positions, tensor.shape[0], tensor.shape[1]).to(tensor.device)
    return tensor[torch.arange(tensor.shape[0], device=tensor.device)[:, None], index]


def estimate_bytes(selections, positions, *, batch_size, hidden_size, element_size, sequence_length=None):
    """Bytes a Capture will copy for one forward pass, computed before running it."""
    count = sequence_length if isinstance(positions, str) else len(positions[0]) if per_row(positions) else len(positions)
    if count is None:
        raise ValueError("sequence_length is required to estimate an 'all' position selection")
    return len(selections) * batch_size * count * hidden_size * element_size


def capture_metadata(runtime, run, phase, token_ids, position_offset=0):
    """Standard metadata for captures made with a loaded Runtime."""
    revision = runtime.provenance.get("download_revision") or runtime.config.get("revision")
    return {"run": run, "phase": phase, "token_ids": token_ids, "position_offset": position_offset,
            "model_revision": revision, "model_id": runtime.config.get("model_id")}


class Capture:
    """Capture explicit token positions with a hard aggregate byte budget.

    Each call creates a detached CPU copy, preventing later cache mutation from
    changing an observation. Metadata is JSON-friendly; tensor data is separate.
    ``token_ids`` (``[batch][token]`` or one sequence) are reduced to the
    selected positions; ``position_offset`` converts input-relative positions
    to absolute sequence positions for cached decode steps.
    """
    def __init__(self, model, selections, *, positions, metadata=None, max_bytes=8 * 1024 * 1024,
                 expected=None, keep_tensors=True):
        if not positions:
            raise ValueError("Explicit token positions are required")
        self.model, self.selections = model, list(selections)
        self.positions = positions if isinstance(positions, str) else \
            [list(p) for p in positions] if per_row(positions) else tuple(positions)
        metadata = dict(metadata or {})
        for key in ("run", "phase", "token_ids", "model_revision"):
            metadata.setdefault(key, None)
        self.metadata, self.max_bytes, self.keep_tensors = metadata, max_bytes, keep_tensors
        self.records, self.bytes_used, self.handles = [], 0, []
        if expected is not None:
            self.estimated_bytes = estimate_bytes(self.selections, self.positions, **expected)
            if self.estimated_bytes > max_bytes:
                raise MemoryError(f"Capture estimate {self.estimated_bytes} bytes exceeds budget {max_bytes}")
        else:
            self.estimated_bytes = None

    def _selected_token_ids(self, index):
        ids = self.metadata.get("token_ids")
        if ids is None:
            return None
        rows = ids if ids and isinstance(ids[0], list) else [ids] * len(index)
        offset = self.metadata.get("position_offset", 0)
        return [[row[offset + p] if offset + p < len(row) else None for p in positions]
                for row, positions in zip(rows, index.tolist())]

    def _record(self, site, value):
        value = first_tensor(value)
        if value.ndim != 3:
            raise ValueError(f"Expected [batch, token, hidden], got {tuple(value.shape)}")
        index = row_indices(self.positions, value.shape[0], value.shape[1])
        size = value.shape[0] * index.shape[1] * value.shape[-1] * value.element_size()
        if self.bytes_used + size > self.max_bytes:
            raise MemoryError(f"Capture requires {self.bytes_used + size} bytes, budget is {self.max_bytes}")
        rows = torch.arange(value.shape[0], device=value.device)[:, None]
        data = value[rows, index.to(value.device)].detach().to(device="cpu", copy=True)
        indices = index[0].tolist() if not per_row(self.positions) else index.tolist()
        self.bytes_used += size
        stats = data.float()
        offset = self.metadata.get("position_offset", 0)
        absolute = [[offset + p for p in row] for row in indices] if per_row(self.positions) else [offset + p for p in indices]
        record = {**{k: v for k, v in self.metadata.items() if k != "token_ids"},
                  "site": site.name, "layer": site.layer, "module_path": site.path, "boundary": site.boundary,
                  "positions": absolute, "input_positions": indices,
                  "token_ids": self._selected_token_ids(index), "axes": list(site.axes),
                  "source_shape": list(value.shape), "shape": list(data.shape),
                  "dtype": str(data.dtype), "bytes": size,
                  "mean": stats.mean().item(), "std": stats.std(unbiased=False).item(),
                  "min": stats.min().item(), "max": stats.max().item(),
                  "slice": stats.flatten()[:8].tolist()}
        if self.keep_tensors:
            record["tensor"] = data
        self.records.append(record)

    def tensor(self, layer, site, index=0):
        """Return the ``index``-th captured tensor for a (layer, site) pair."""
        matches = [r for r in self.records if r["layer"] == layer and r["site"] == site]
        return matches[index]["tensor"]

    def __enter__(self):
        try:
            for layer, name in self.selections:
                site = resolve_site(self.model, layer, name)
                if site.boundary == "input":
                    handle = site.module.register_forward_pre_hook(lambda m, args, s=site: self._record(s, args[0]))
                else:
                    handle = site.module.register_forward_hook(lambda m, args, out, s=site: self._record(s, out))
                self.handles.append(handle)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False
