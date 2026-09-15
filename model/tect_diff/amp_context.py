"""Keep inference casts from removing gradients from a later training pass."""
from contextlib import contextmanager

import torch


@contextmanager
def self_condition_no_grad():
    # PyTorch 2.1 caches no-grad casts of trainable weights in the outer AMP
    # context. Reusing them in the task pass silently drops their gradients.
    # Change only cache use; preserve the caller's device, precision and RNG.
    cache_enabled = torch.is_autocast_cache_enabled()
    try:
        torch.set_autocast_cache_enabled(False)
        with torch.no_grad():
            yield
    finally:
        torch.set_autocast_cache_enabled(cache_enabled)
