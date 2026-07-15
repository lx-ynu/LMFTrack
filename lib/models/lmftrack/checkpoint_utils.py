from __future__ import annotations

from collections import OrderedDict
from typing import Mapping


LEGACY_KEY_REPLACEMENTS = (
    ("backbone.language_prompt.", "backbone.cspg."),
    ("backbone.cstnet_fusion_layers.", "backbone.fusion_blocks."),
    (".cfm.", ".mcsaii."),
    (".gim1.", ".template_mixing."),
    (".gim2.", ".search_mixing."),
    (".sfm.", ".mmscf."),
    (".mcsaii.gc.gvp.", ".mcsaii.nlca.global_pool."),
    (".mcsaii.gc.", ".mcsaii.nlca."),
    (".mcsaii.scsa.", ".mcsaii.smsa."),
    (".tksca.", ".sparse_cross_attention."),
    (".template_mixing.mlp1.", ".template_mixing.mlp."),
    (".search_mixing.mlp1.", ".search_mixing.mlp."),
    (".mmscf.lpu.conv.", ".mmscf.lpu.depthwise_conv."),
    (".lpu.dwconv.", ".lpu.depthwise_conv."),
    (".cfn.conv33.", ".cfn.conv3."),
    (".cfn.bn33.", ".cfn.bn3."),
    (".cfn.conv11.", ".cfn.conv1."),
    (".cfn.bn11.", ".cfn.bn1."),
    (".cfn.act.", ".cfn.activation."),
    ("tbsi_fuse_search.", "search_fusion."),
)

# These parameters belonged to modules instantiated by the inherited code but
# never used in its forward path. They are intentionally omitted from the
# cleaned implementation.
LEGACY_UNUSED_KEY_PARTS = (
    ".mcsaii.lsa.",
    ".mcsaii.smsa.norm.",
)



def remap_legacy_state_dict(state_dict: Mapping[str, object]):
    """Map checkpoints produced by the earlier TBSI-named code to the cleaned LMFTrack names."""
    remapped = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for old, new in LEGACY_KEY_REPLACEMENTS:
            new_key = new_key.replace(old, new)
        if any(part in new_key for part in LEGACY_UNUSED_KEY_PARTS):
            continue
        remapped[new_key] = value
    return remapped
