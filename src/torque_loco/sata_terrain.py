# src/torque_loco/sata_terrain.py
"""SATA-faithful terrain for the Isaac Lab Go2 migration.

SATA (legged_gym go2_torque_config.py) trains on a trimesh terrain with
``terrain_proportions = [0.2, 0.8, 0, 0, 0]`` and ``curriculum = False``:

    * smooth slope (20%):  ``pyramid_sloped(slope = difficulty * 0.1)``        (legged_gym terrain.py:133)
    * rough  slope (80%):  the SAME gentle slope PLUS ``random_uniform`` noise   (terrain.py:137-139)
                           ``min=-0.06, max=0.06, step=0.005, downsampled_scale=0.2``
    * no stairs / discrete / stepping-stones (their proportions are 0).

Isaac Lab's height-field primitives keep slope and noise as SEPARATE sub-terrains, so to
reproduce a "rough slope" (a slope WITH roughness on top, in one cell) we author one custom
height-field function that sums the two — mirroring legged_gym, which applies
``random_uniform_terrain`` on top of ``pyramid_sloped_terrain`` in the same SubTerrain.

Cross-engine note: the two engines' terrain generators differ, so the height fields cannot be
bit-identical. We match the terrain TYPE and difficulty distribution (gentle slopes 0..0.1 rad +
-0.06..0.06 m roughness, no stairs, no curriculum), which is what governs the torque envelope.
"""
from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import scipy.interpolate as interpolate

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


@height_field_to_mesh
def sata_rough_slope_terrain(difficulty: float, cfg: "SataRoughSlopeTerrainCfg") -> np.ndarray:
    """A gentle pyramid slope (like SATA's smooth slope) with uniform noise summed on top.

    Combines isaaclab's ``pyramid_sloped_terrain`` (slope) and ``random_uniform_terrain`` (noise)
    into a single height field, replicating legged_gym's "rough slope" cell.
    Returns the discretized height field (width, length) in vertical-scale units (int16).
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    # --- pyramid slope (mirrors isaaclab.terrains.height_field.hf_terrains.pyramid_sloped_terrain) ---
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)
    xx, yy = np.meshgrid(np.arange(0, width_pixels), np.arange(0, length_pixels), sparse=True)
    xx = ((center_x - np.abs(center_x - xx)) / center_x).reshape(width_pixels, 1)
    yy = ((center_y - np.abs(center_y - yy)) / center_y).reshape(1, length_pixels)
    hf_slope = height_max * xx * yy
    # flat platform at the center
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    z_pf = hf_slope[width_pixels // 2 - platform_width, length_pixels // 2 - platform_width]
    hf_slope = np.clip(hf_slope, min(0, z_pf), max(0, z_pf))

    # --- uniform noise (mirrors random_uniform_terrain), summed on top like legged_gym ---
    ds = cfg.downsampled_scale if cfg.downsampled_scale is not None else cfg.horizontal_scale
    width_ds = int(cfg.size[0] / ds)
    length_ds = int(cfg.size[1] / ds)
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max_n = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)
    height_range = np.arange(height_min, height_max_n + height_step, height_step)
    noise_ds = np.random.choice(height_range, size=(width_ds, length_ds))
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_ds)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_ds)
    func = interpolate.RectBivariateSpline(x, y, noise_ds)
    x_up = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_up = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    hf_noise = func(x_up, y_up)

    return np.rint(hf_slope + hf_noise).astype(np.int16)


@configclass
class SataRoughSlopeTerrainCfg(HfTerrainBaseCfg):
    """A SATA "rough slope": gentle pyramid slope + uniform roughness in one cell."""

    function = sata_rough_slope_terrain

    slope_range: tuple[float, float] = MISSING
    """Pyramid slope (rad), interpolated by difficulty. SATA uses slope = difficulty * 0.1."""
    platform_width: float = 3.0
    """Flat platform width at the centre (m). SATA uses platform_size=3.0."""
    inverted: bool = False
    """If True the slope descends from the centre (legged_gym flips sign for half the cells)."""
    noise_range: tuple[float, float] = (-0.06, 0.06)
    """Uniform roughness range (m). SATA: min=-0.06, max=0.06."""
    noise_step: float = 0.005
    """Roughness quantisation (m). SATA: step=0.005."""
    downsampled_scale: float = 0.2
    """Roughness sampling resolution (m). SATA: downsampled_scale=0.2."""


# SATA terrain: 0.2 smooth slope (0.1 up + 0.1 down) + 0.8 rough slope (0.4 up + 0.4 down).
# Gentle slopes (0..0.1 rad, == SATA's slope = difficulty*0.1) + -0.06..0.06 m roughness.
# curriculum=False, num_rows=10, num_cols=20, h/v scale 0.1/0.005 — all matching SATA.
SATA_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=False,
    use_cache=False,
    sub_terrains={
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.1), platform_width=3.0, border_width=0.25
        ),
        "smooth_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.1), platform_width=3.0, border_width=0.25
        ),
        "rough_slope": SataRoughSlopeTerrainCfg(
            proportion=0.4, slope_range=(0.0, 0.1), platform_width=3.0, inverted=False
        ),
        "rough_slope_inv": SataRoughSlopeTerrainCfg(
            proportion=0.4, slope_range=(0.0, 0.1), platform_width=3.0, inverted=True
        ),
    },
)
