"""Bounded, exception-safe capture at semantic decoder sites."""
from .sites import resolve_site


def first_tensor(value):
    return value[0] if isinstance(value, tuple) else value


def replace_first(value, tensor):
    return (tensor, *value[1:]) if isinstance(value, tuple) else tensor


def select_positions(tensor, positions):
    if tensor.ndim != 3:
        raise ValueError(f"Expected [batch, token, hidden], got {tuple(tensor.shape)}")
    indices = [p if p >= 0 else tensor.shape[1] + p for p in positions]
    if any(p < 0 or p >= tensor.shape[1] for p in indices):
        raise IndexError(f"Token position {positions} outside sequence length {tensor.shape[1]}")
    return tensor[:, indices, :]


class Capture:
    """Capture explicit token positions with a hard aggregate byte budget.

    Each call creates a detached CPU copy, preventing later cache mutation from
    changing an observation. Metadata is JSON-friendly; tensor data is separate.
    """
    def __init__(self, model, selections, *, positions, metadata=None, max_bytes=8 * 1024 * 1024):
        if not positions:
            raise ValueError("Explicit token positions are required")
        self.model, self.selections, self.positions = model, selections, tuple(positions)
        self.metadata, self.max_bytes = dict(metadata or {}), max_bytes
        self.records, self.bytes_used, self.handles = [], 0, []

    def _record(self, site, value):
        value = first_tensor(value)
        if value.ndim != 3:
            raise ValueError(f"Expected [batch, token, hidden], got {tuple(value.shape)}")
        size = value.shape[0] * len(self.positions) * value.shape[-1] * value.element_size()
        if self.bytes_used + size > self.max_bytes:
            raise MemoryError(f"Capture requires {self.bytes_used + size} bytes, budget is {self.max_bytes}")
        selected = select_positions(value, self.positions)
        data = selected.detach().to(device="cpu", copy=True)
        self.bytes_used += size
        stats = data.float()
        self.records.append({**self.metadata, "site": site.name, "layer": site.layer,
            "module_path": site.path, "boundary": site.boundary,
            "positions": list(self.positions), "axes": list(site.axes),
            "source_shape": list(value.shape), "shape": list(data.shape),
            "dtype": str(data.dtype), "estimated_bytes": size, "bytes": size,
            "mean": stats.mean().item(), "std": stats.std(unbiased=False).item(),
            "min": stats.min().item(), "max": stats.max().item(),
            "slice": stats.flatten()[:8].tolist(), "tensor": data})

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
