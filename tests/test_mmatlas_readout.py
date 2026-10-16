"""Token layout of the MultimodalAtlas "activ" readout.

Each token carries its own grid coordinates as features, so the readout can be
checked position by position. Tokens are windowed the way Pillar0-ChestCT's
MultiModalAtlas._build_multiscale_tokens does it, for the chest CT config
(image_size 256^3, patch_size (8, 8, 4), merge_ratio and local2global (4, 4, 4)).
"""

import pytest
import torch
from einops import rearrange

from pillar.models.backbones.mmatlas import tokens_to_grid

RES = [32, 32, 64]
MERGE = (4, 4, 4)
# prepare_multiscale_layout output for the chest CT config
LAYOUT = [
    {"grid_size": [8, 8, 16], "window_dims": [4, 4, 4]},
    {"grid_size": [2, 2, 4], "window_dims": [4, 4, 4]},
    {"grid_size": [1, 1, 1], "window_dims": [2, 2, 4]},
]


def coords(shape):
    """(1, 3, *shape) tensor whose features are each voxel's (a0, a1, a2) index."""
    grids = torch.meshgrid(*[torch.arange(n) for n in shape], indexing="ij")
    return torch.stack(grids).float()[None]


def atlas_tokens(x_BCDHW):
    """Window one scale like _build_multiscale_tokens, then flatten like MultimodalAtlas.forward."""
    b, _, d, h, w = x_BCDHW.shape
    if (d, h, w) == MERGE or d < MERGE[0] or h < MERGE[1] or w < MERGE[2]:
        return rearrange(x_BCDHW, "b c d h w -> b (d h w) c")
    x = rearrange(x_BCDHW, "b c (d m0) (h m1) (w m2) -> (b d h w) (m0 m1 m2) c",
                  m0=MERGE[0], m1=MERGE[1], m2=MERGE[2])
    return rearrange(x, "(b nw) k c -> b (nw k) c", b=b)


SCALE_SHAPES = [(32, 32, 64), (8, 8, 16), (2, 2, 4)]
FEATS = [atlas_tokens(coords(s)) for s in SCALE_SHAPES]


def test_spatial_layout_puts_every_token_where_it_was():
    activ = tokens_to_grid(FEATS, LAYOUT, RES, "spatial")
    assert activ.shape == (1, 9, *RES)
    for k, shape in enumerate(SCALE_SHAPES):
        expected = torch.nn.functional.interpolate(coords(shape), size=RES, mode="nearest")
        assert torch.equal(activ[:, 3 * k : 3 * k + 3], expected), f"scale {k}"


def test_legacy_layout_is_unchanged():
    """The default reproduces the readout the released checkpoints were trained with."""
    expected = []
    for layout, scale in zip(LAYOUT, FEATS):
        m0, m1, m2 = layout["window_dims"]
        g0, g1, g2 = layout["grid_size"]
        scale = rearrange(scale, "b (g2 g0 g1 m2 m0 m1) c -> b c (g2 m2) (g0 m0) (g1 m1)",
                          g2=g2, g1=g1, g0=g0, m2=m2, m1=m1, m0=m0)
        expected.append(torch.nn.functional.interpolate(scale, size=RES, mode="nearest"))
    assert torch.equal(tokens_to_grid(FEATS, LAYOUT, RES), torch.cat(expected, dim=1))


def test_legacy_layout_drops_half_of_the_fine_tokens():
    fine = tokens_to_grid(FEATS, LAYOUT, RES, "legacy")[0, :3].long()
    kept = torch.unique(fine.reshape(3, -1).T, dim=0)
    assert kept.shape[0] == 32 * 32 * 64 // 2
    assert torch.unique(fine[0]).tolist() == list(range(0, 32, 2))


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError):
        tokens_to_grid(FEATS, LAYOUT, RES, "zyx")
