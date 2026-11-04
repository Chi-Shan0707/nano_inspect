"""Scoped activation patching and additive steering at explicit token positions."""
import torch
from .sites import resolve_site
from .capture import first_tensor, replace_first, normalize_positions


class Intervention:
    """Replace (``patch``) or shift (``h' = h + alpha * direction``) one site.

    Positions are relative to the current forward input (``[0]`` or ``[-1]``
    addresses the single token of a cached decode step); ``"all"`` selects
    every input position. A patch has shape ``[batch, positions, hidden]``.
    ``calls`` counts hook invocations so callers can confirm the declared site
    actually executed.
    """
    def __init__(self, model, layer, site, positions, *, patch=None, direction=None, alpha=0.0):
        if not positions:
            raise ValueError("Explicit token positions are required")
        if (patch is None) == (direction is None):
            raise ValueError("Specify exactly one of patch and direction")
        self.site = resolve_site(model, layer, site)
        self.positions = positions if isinstance(positions, str) else tuple(positions)
        self.patch, self.direction, self.alpha = patch, direction, alpha
        self.handle, self.calls = None, 0

    def _apply(self, value):
        original = first_tensor(value)
        if original.ndim != 3:
            raise ValueError(f"Expected [batch, token, hidden], got {tuple(original.shape)}")
        indices = normalize_positions(self.positions, original.shape[1])
        self.calls += 1
        result = original.clone()
        if self.patch is not None:
            replacement = self.patch.to(device=original.device, dtype=original.dtype)
            expected = (original.shape[0], len(indices), original.shape[-1])
            if tuple(replacement.shape) != expected:
                raise ValueError(f"Patch shape {tuple(replacement.shape)} must equal selected shape {expected}")
            result[:, indices, :] = replacement
        else:
            direction = self.direction.to(device=original.device, dtype=torch.float32)
            if tuple(direction.shape) != (original.shape[-1],):
                raise ValueError("Direction must be a single hidden-width vector")
            if self.alpha == 0:
                return value  # exact identity: zero strength must not even round-trip dtypes
            shifted = original[:, indices, :].float() + self.alpha * direction
            result[:, indices, :] = shifted.to(original.dtype)
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
    """Mean aligned positive-minus-negative residual; optionally unit L2 norm.

    Returns ``(direction, raw_norm)`` where ``raw_norm`` is the L2 norm of the
    unnormalized mean difference, in residual-stream units.
    """
    if positive.shape != negative.shape or positive.ndim != 2 or positive.shape[0] == 0:
        raise ValueError("Aligned examples must have equal nonempty [examples, hidden] shapes")
    direction = (positive.float() - negative.float()).mean(dim=0)
    norm = direction.norm()
    if normalize:
        if norm.item() == 0:
            raise ValueError("Contrastive direction has zero norm")
        direction = direction / norm
    return direction, norm.item()


def random_direction(width, seed, norm=1.0):
    """Gaussian direction with an exact L2 norm, reproducible from ``seed``."""
    generator = torch.Generator().manual_seed(int(seed))
    vector = torch.randn(width, generator=generator, dtype=torch.float32)
    return vector / vector.norm() * norm


def hook_count(model):
    """Number of registered forward/pre hooks across all modules (zero after cleanup)."""
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
