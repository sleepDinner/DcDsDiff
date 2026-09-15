"""Versioned normalization for freshly initialized TECT task networks."""

from torch import nn


MAIN_ARCHITECTURE_V1 = "tect-diff-full-v1"
MAIN_ARCHITECTURE_GN8 = "tect-diff-full-v2-main-gn8"

# Preserve these module names: MMCV ConvModule resolves its normalization via
# norm_name == 'bn', independently of the normalization module's Python type.
TASK_BATCHNORMS = {
    **{f"mmff{scale}.reduce{stream}.bn": 256
       for scale in range(1, 5) for stream in (1, 2)},
    "msff.fusion.bn": 256,
    "mask.down2.0.bn": 256,
    "mask.down2.2.bn": 256,
    "mask.up2.0.bn": 256,
    "mask.up2.2.bn": 64,
    "mask.up2.4.bn": 32,
    "mask.pred2.0.bn": 32,
}


def require_main_normalization(normalization: str, architecture_version: str):
    versions = {"batchnorm": MAIN_ARCHITECTURE_V1, "groupnorm8": MAIN_ARCHITECTURE_GN8}
    if not isinstance(normalization, str) or normalization not in versions:
        raise ValueError(f"Unknown task normalization: {normalization!r}")
    if not isinstance(architecture_version, str) or architecture_version not in versions.values():
        raise ValueError(f"Unknown task architecture: {architecture_version!r}")
    if architecture_version != versions[normalization]:
        raise ValueError("Task architecture and normalization do not match")


def configure_main_normalization(network: nn.Module, normalization: str,
                                 architecture_version: str):
    """Validate all 15 original BN sites, then optionally replace them with GN8.

    Called only by TECTNetwork.__init__, before optimizer creation or checkpoint
    loading. This is an architecture revision, not a trained-state conversion.
    """
    require_main_normalization(normalization, architecture_version)
    found = {name: module for name, module in network.named_modules()
             if isinstance(module, nn.modules.batchnorm._BatchNorm)}
    if set(found) != set(TASK_BATCHNORMS):
        raise RuntimeError("TECT task BatchNorm inventory changed: "
                           f"missing={sorted(set(TASK_BATCHNORMS) - set(found))}, "
                           f"unexpected={sorted(set(found) - set(TASK_BATCHNORMS))}")
    # Validate the entire inventory before modifying any module.
    for name, channels in TASK_BATCHNORMS.items():
        layer = found[name]
        if (type(layer) is not nn.BatchNorm2d or layer.num_features != channels
                or layer.eps != 1e-5 or not layer.affine or channels % 8):
            raise RuntimeError(f"TECT task BatchNorm contract changed at {name}")
    if normalization == "groupnorm8":
        for name, channels in TASK_BATCHNORMS.items():
            parent_name, attribute = name.rsplit(".", 1)
            layer = nn.GroupNorm(8, channels, eps=1e-5, affine=True)
            # Fresh BN and GN both initialize affine weight=1, bias=0; GN
            # initialization consumes no RNG and introduces no running buffers.
            setattr(network.get_submodule(parent_name), attribute, layer)
        if any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
               for layer in network.modules()):
            raise RuntimeError("TECT GroupNorm task network still contains BatchNorm")
