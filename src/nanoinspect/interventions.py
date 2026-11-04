"""Scoped activation patching and additive steering at explicit token positions."""
import torch
from .sites import resolve_site
from .capture import first_tensor, replace_first, select_positions


class Intervention:
    def __init__(self, model, layer, site, positions, *, patch=None, direction=None, alpha=0.0):
        if not positions:
            raise ValueError("Explicit token positions are required")
        if (patch is None) == (direction is None):
            raise ValueError("Specify exactly one of patch and direction")
        self.site = resolve_site(model, layer, site)
        self.positions, self.patch, self.direction, self.alpha = tuple(positions), patch, direction, alpha
        self.handle = None

    def _apply(self, value):
        original = first_tensor(value)
        selected = select_positions(original, self.positions)
        replacement = self.patch if self.patch is not None else self.direction
        replacement = replacement.to(device=original.device, dtype=original.dtype)
        if self.patch is not None and tuple(replacement.shape) != tuple(selected.shape):
            raise ValueError(f"Patch shape {tuple(replacement.shape)} must equal selected shape {tuple(selected.shape)}")
        if self.direction is not None and tuple(replacement.shape) != (original.shape[-1],):
            raise ValueError("Direction must be a single hidden-width vector")
        result = original.clone()
        indices = [p if p >= 0 else original.shape[1] + p for p in self.positions]
        result[:, indices, :] = replacement if self.patch is not None else selected + self.alpha * replacement
        return replace_first(value, result)

    def __enter__(self):
        if self.site.boundary == "input":
            def pre_hook(module, args):
                return (self._apply(args[0]), *args[1:])
            self.handle = self.site.module.register_forward_pre_hook(pre_hook)
        else:
            self.handle = self.site.module.register_forward_hook(lambda module, args, out: self._apply(out))
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


def contrastive_direction(positive, negative, normalize=True):
    """Mean aligned positive-minus-negative residual; optionally unit L2 norm."""
    if positive.shape != negative.shape or positive.ndim != 2 or positive.shape[0] == 0:
        raise ValueError("Aligned examples must have equal nonempty [examples, hidden] shapes")
    direction = (positive.float() - negative.float()).mean(dim=0)
    norm = direction.norm()
    if normalize:
        if norm.item() == 0:
            raise ValueError("Contrastive direction has zero norm")
        direction = direction / norm
    return direction, norm.item()
