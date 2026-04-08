"""Triple-template localization simulation.

Observation map:
    Simulates the live image source seen by the UAV. Three neighboring crops
    are extracted from this map.

Reference map:
    The map searched independently by each crop. The final predicted position
    is estimated from the intersection of the matched boxes.
"""

import concurrent.futures
import json
import math
import os
import random
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import cv2
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyproj import Transformer
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.models import load_model

UP_KEYS = (ord("w"), ord("W"), 82, 2490368, 65362)
DOWN_KEYS = (ord("s"), ord("S"), 84, 2621440, 65364)
LEFT_KEYS = (ord("a"), ord("A"), 81, 2424832, 65361)
RIGHT_KEYS = (ord("d"), ord("D"), 83, 2555904, 65363)
ROTATE_LEFT_KEYS = (ord("q"), ord("Q"))
ROTATE_RIGHT_KEYS = (ord("e"), ord("E"))
ALTITUDE_UP_KEYS = (ord("+"), ord("="), 43, 61, 107)
ALTITUDE_DOWN_KEYS = (ord("-"), ord("_"), 45, 95, 109)
EXIT_KEYS = (27, ord("x"), ord("X"))
COMPASS_LABELS = ("K", "KD", "D", "GD", "G", "GB", "B", "KB")
TEMPLATE_COLORS = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
)
UI_COLORS = {
    "panel_bg": (30, 33, 42),
    "panel_border": (90, 96, 112),
    "btn_on": (56, 124, 245),
    "btn_off": (58, 60, 70),
    "btn_hover_on": (78, 146, 255),
    "btn_hover_off": (80, 84, 96),
    "toggle_on": (76, 175, 80),
    "toggle_off": (120, 120, 130),
    "toggle_knob": (255, 255, 255),
    "text_primary": (245, 247, 252),
    "text_shadow": (0, 0, 0),
    "accent": (66, 133, 244),
    "header_bg": (45, 48, 65),
    "collapse_btn": (55, 60, 80),
    "collapse_hover": (80, 85, 110),
}


@dataclass(frozen=True)
class SimulationConfig:
    scenario_mode: str = "irtifa"  # "normal" veya "irtifa"
    reference_map_path: Path = Path(
        "haritalar/ana_harita_urgup_30_cm__GPU_model_f32_k3_epoch_00001_sigmoid_(1_ 1)_06_10_2022_.h5.jpg_resized.jpg_geo.tif_geo.tif_r.tif"
    )
    observation_map_path: Path = Path("parcalar/urgup_bingmap_30cm_utm.tif")
    observation_georef_path: Path = Path("parcalar/urgup_bingmap_30cm_utm.tif")
    observation_grid_georef_path: Optional[Path] = Path(
        "haritalar/ana_harita_urgup_30_cm__GPU_model_f32_k3_epoch_00001_sigmoid_(1_ 1)_06_10_2022_.h5.jpg_resized.jpg_geo.tif_geo.tif_r.tif"
    )
    dem_path: Path = Path("ana_harita_urgup_30_cm_utm_elevation.tif")
    model_path: Path = Path("GPU_model_f32_k3_epoch_00001_sigmoid_(1_ 1)_06_10_2022_.h5")
    sample_window_size: int = 544
    model_input_size: int = 544
    crop_margin: int = 16
    template_size: int = 512
    template_offset: int = 100
    initial_row: int = 2500
    initial_col: int = 2500
    random_start: bool = True
    random_start_middle_band_ratio: float = 0.50
    step_size: int = 250
    initial_heading_degrees: float = 0.0
    rotation_step_degrees: float = 15.0
    initial_altitude_agl_m: float = 110.0
    altitude_step_m: float = 10.0
    min_altitude_agl_m: float = 30.0
    max_altitude_agl_m: float = 250.0
    minimum_patch_agl_m: float = 5.0
    reference_map_gsd_cm_per_px: float = 29.85
    camera_sensor_width_mm: float = 13.2
    camera_focal_length_mm: float = 8.8
    virtual_camera_width_px: int = 544
    align_observation_to_reference_grid: bool = True
    display_size: Tuple[int, int] = (1000, 1000)
    left_panel_width_ratio: float = 0.38
    match_method: int = cv2.TM_CCOEFF_NORMED
    use_parallel_matching: bool = True
    use_pyramid_matching: bool = True
    coarse_scale: float = 0.5
    roi_pad_factor: float = 0.4
    base_search_window_size: int = 2048
    max_search_window_size: int = 15000
    search_window_growth_step: int = 100
    search_window_failure_growth: int = 500
    triplet_alignment_tolerance_px: float = 45.0
    global_refresh_interval: int = 0
    dashboard_background_color: Tuple[int, int, int] = (18, 18, 24)
    panel_background_color: Tuple[int, int, int] = (32, 36, 42)
    panel_border_color: Tuple[int, int, int] = (78, 84, 92)
    panel_title_color: Tuple[int, int, int] = (230, 230, 230)
    actual_path_color: Tuple[int, int, int] = (0, 255, 120)
    predicted_path_color: Tuple[int, int, int] = (0, 170, 255)
    actual_intersection_color: Tuple[int, int, int] = (0, 204, 0)
    predicted_intersection_color: Tuple[int, int, int] = (0, 215, 255)
    error_line_color: Tuple[int, int, int] = (255, 255, 0)
    dashboard_window_name: str = "Dashboard"
    observation_panel_title: str = "Gozlem Alani"
    template_panel_title: str = "Merkez Model Ciktisi"
    reference_panel_title: str = "Referans Harita"
    panel_padding: int = 20
    panel_gap: int = 20
    panel_inner_padding: int = 12
    panel_title_height: int = 38
    hud_font_scale: float = 0.70
    hud_font_thickness: int = 2
    path_history_limit: int = 120
    observation_context_margin: int = 120
    template_strip_tile_size: int = 180
    template_strip_gap: int = 12
    rectangle_thickness: int = 3
    search_window_color: Tuple[int, int, int] = (0, 165, 255)
    heading_indicator_color: Tuple[int, int, int] = (255, 220, 0)
    ui_buttons_enabled: bool = True
    ui_button_font_scale: float = 0.82
    ui_button_thickness: int = 2
    ui_button_scale: float = 0.40
    show_info_panel: bool = True
    show_trajectory: bool = True
    show_roi_frame: bool = True
    show_tm_boxes: bool = True
    show_heading_arrow: bool = True
    show_observation_boxes: bool = True
    reference_viewport_base_size: int = 6000
    reference_viewport_padding: int = 600
    reference_viewport_search_padding: int = 320
    reference_viewport_search_min_size: int = 4200
    diagnostic_benchmark_enabled: bool = False
    diagnostic_benchmark_only: bool = False
    diagnostic_output_dir: Path = Path("diagnostics")
    diagnostic_tile_size: int = 256
    diagnostic_benchmark_points: Tuple[Tuple[int, int], ...] = (
        (12000, 15000),
        (8000, 10000),
        (16000, 22000),
        (6000, 24000),
        (18000, 12000),
    )


@dataclass(frozen=True)
class ReferencePreviewState:
    panel_rect: Tuple[int, int, int, int]
    paste_x: int
    paste_y: int
    preview_width: int
    preview_height: int
    scale_x: float
    scale_y: float
    viewport_left: int
    viewport_top: int
    viewport_width: int
    viewport_height: int
    base_preview: np.ndarray


@dataclass
class TerrainContext:
    dem_dataset: object
    observation_dataset: object
    observation_to_dem_transformer: object
    resized_observation_shape: Tuple[int, int]


@dataclass(frozen=True)
class AltitudeSimulationState:
    altitude_agl_m: float
    altitude_msl_m: float
    center_ground_elevation_m: float
    patch_ground_elevations_m: Tuple[float, float, float]
    patch_agl_m: Tuple[float, float, float]
    patch_scale_factors: Tuple[float, float, float]
    center_gsd_cm_per_px: float


class _CompatConv2DTranspose(Conv2DTranspose):
    @classmethod
    def from_config(cls, config):
        compat_config = dict(config or {})
        compat_config.pop("groups", None)
        return super().from_config(compat_config)


def load_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), 0)
    if image is None:
        raise FileNotFoundError("Image could not be loaded: %s" % path)
    return image


def is_georaster_path(path: Path) -> bool:
    return path.suffix.lower() in (".tif", ".tiff")


def read_raster_dataset_as_grayscale(dataset: object) -> np.ndarray:
    if dataset.count <= 0:
        raise ValueError("Raster has no bands: %s" % getattr(dataset, "name", "<unknown>"))

    if dataset.count == 1:
        grayscale = dataset.read(1)
    else:
        rgb_band_count = min(3, dataset.count)
        rgb = np.moveaxis(dataset.read(list(range(1, rgb_band_count + 1))), 0, -1)
        if rgb.shape[2] == 1:
            grayscale = rgb[:, :, 0]
        else:
            grayscale = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)

    if grayscale.dtype == np.uint8:
        return grayscale
    return cv2.normalize(
        grayscale,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )


def load_grayscale_raster(path: Path) -> np.ndarray:
    with rio.open(str(path)) as dataset:
        return read_raster_dataset_as_grayscale(dataset)


def load_observation_aligned_to_reference_grid(
    observation_path: Path,
    reference_path: Path,
) -> np.ndarray:
    with rio.open(str(observation_path)) as observation_dataset, rio.open(
        str(reference_path)
    ) as reference_dataset:
        source_gray = read_raster_dataset_as_grayscale(observation_dataset).astype(
            np.float32
        )
        aligned = np.zeros(
            (reference_dataset.height, reference_dataset.width),
            dtype=np.float32,
        )
        reproject(
            source=source_gray,
            destination=aligned,
            src_transform=observation_dataset.transform,
            src_crs=observation_dataset.crs,
            dst_transform=reference_dataset.transform,
            dst_crs=reference_dataset.crs,
            resampling=Resampling.nearest,
            dst_nodata=0.0,
        )
    return np.clip(aligned, 0.0, 255.0).astype(np.uint8)


def load_model_compat(model_path: Path):
    try:
        return load_model(str(model_path), compile=False)
    except TypeError as exc:
        if "Conv2DTranspose" in str(exc) and "groups" in str(exc):
            return load_model(
                str(model_path),
                compile=False,
                custom_objects={"Conv2DTranspose": _CompatConv2DTranspose},
            )
        raise


def get_output_template_size(config: SimulationConfig) -> int:
    return config.model_input_size - (2 * config.crop_margin)


def validate_config(config: SimulationConfig) -> None:
    normalize_scenario_mode(config.scenario_mode)
    output_template_size = get_output_template_size(config)
    if output_template_size <= 0:
        raise ValueError("crop_margin model_input_size icin fazla buyuk.")
    if output_template_size != config.template_size:
        raise ValueError(
            "template_size=%d ama model cikti boyutu=%d."
            % (config.template_size, output_template_size)
        )
    if config.sample_window_size <= 0 or config.model_input_size <= 0:
        raise ValueError("Model pencere boyutlari pozitif olmali.")


def normalize_heading_degrees(heading_degrees: float) -> float:
    return float(heading_degrees % 360.0)


def rotate_image_offset(
    delta_x: float,
    delta_y: float,
    angle_degrees: float,
) -> Tuple[float, float]:
    angle_radians = math.radians(normalize_heading_degrees(angle_degrees))
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    return (
        (delta_x * cos_angle) - (delta_y * sin_angle),
        (delta_x * sin_angle) + (delta_y * cos_angle),
    )


def get_heading_vector(heading_degrees: float) -> Tuple[float, float]:
    return rotate_image_offset(0.0, -1.0, heading_degrees)


def get_heading_label(heading_degrees: float) -> str:
    normalized_heading = normalize_heading_degrees(heading_degrees)
    direction_index = int(((normalized_heading + 22.5) % 360.0) // 45.0)
    return "%.1f° %s" % (normalized_heading, COMPASS_LABELS[direction_index])


def get_action_label(action: str) -> str:
    action_labels = {
        "forward": "ileri",
        "backward": "geri",
        "strafe_left": "yan-sol",
        "strafe_right": "yan-sag",
        "rotate_left": "don-sol",
        "rotate_right": "don-sag",
        "altitude_up": "irtifa+",
        "altitude_down": "irtifa-",
    }
    return action_labels.get(action, "bekle")


def get_triplet_rotation_margin(config: SimulationConfig) -> int:
    return int(config.template_offset)


def get_scaled_observation_window_size(
    scale_factor: float,
    config: SimulationConfig,
) -> int:
    window_size = int(
        round(config.sample_window_size * max(0.05, float(scale_factor)))
    )
    window_size = max(32, window_size)
    if (window_size % 2) != (config.sample_window_size % 2):
        window_size += 1
    return window_size


def get_rotated_capture_size(config: SimulationConfig) -> int:
    max_scale_factor = 1.0
    if is_altitude_scenario(config):
        max_scale_factor = max(
            1.0,
            compute_scale_factor_for_altitude(config.max_altitude_agl_m, config),
        )
    max_window_size = get_scaled_observation_window_size(
        max_scale_factor * 1.15,
        config,
    )
    capture_size = int(math.ceil(max_window_size * math.sqrt(2.0))) + 2
    if (capture_size % 2) != (config.sample_window_size % 2):
        capture_size += 1
    return capture_size


def get_observation_cursor_limits(
    image_shape: Tuple[int, int],
    config: SimulationConfig,
) -> Tuple[int, int, int]:
    height, width = image_shape
    triplet_margin = get_triplet_rotation_margin(config)
    capture_half = get_rotated_capture_size(config) // 2
    sample_half = config.sample_window_size // 2
    minimum = sample_half + capture_half + triplet_margin
    maximum_row = max(minimum, height + sample_half - capture_half - triplet_margin)
    maximum_col = max(minimum, width + sample_half - capture_half - triplet_margin)
    return minimum, maximum_row, maximum_col


def format_heading_label(heading_degrees: float) -> str:
    normalized_heading = normalize_heading_degrees(heading_degrees)
    direction_index = int(((normalized_heading + 22.5) % 360.0) // 45.0)
    return "%.1f deg %s" % (normalized_heading, COMPASS_LABELS[direction_index])


def normalize_scenario_mode(scenario_mode: str) -> str:
    normalized_mode = str(scenario_mode).strip().lower()
    if normalized_mode in ("normal", "standart"):
        return "normal"
    if normalized_mode in ("irtifa", "altitude", "elevation"):
        return "irtifa"
    raise ValueError(
        "scenario_mode gecersiz: %s. Desteklenen degerler: normal, irtifa."
        % scenario_mode
    )


def is_altitude_scenario(config: SimulationConfig) -> bool:
    return normalize_scenario_mode(config.scenario_mode) == "irtifa"


def get_scenario_label(config: SimulationConfig) -> str:
    return normalize_scenario_mode(config.scenario_mode)


def clamp_altitude_agl(altitude_agl_m: float, config: SimulationConfig) -> float:
    return float(
        min(
            max(altitude_agl_m, config.min_altitude_agl_m),
            config.max_altitude_agl_m,
        )
    )


def compute_virtual_camera_gsd_cm_per_px(
    altitude_agl_m: float,
    config: SimulationConfig,
) -> float:
    effective_altitude = max(float(altitude_agl_m), float(config.minimum_patch_agl_m))
    return (
        float(config.camera_sensor_width_mm)
        * effective_altitude
        * 100.0
        / (float(config.camera_focal_length_mm) * float(config.virtual_camera_width_px))
    )


def compute_scale_factor_for_altitude(
    altitude_agl_m: float,
    config: SimulationConfig,
) -> float:
    return compute_virtual_camera_gsd_cm_per_px(altitude_agl_m, config) / float(
        config.reference_map_gsd_cm_per_px
    )


def build_normal_altitude_state(patch_count: int) -> AltitudeSimulationState:
    zero_tuple = tuple(0.0 for _ in range(patch_count))
    scale_tuple = tuple(1.0 for _ in range(patch_count))
    return AltitudeSimulationState(
        altitude_agl_m=0.0,
        altitude_msl_m=0.0,
        center_ground_elevation_m=0.0,
        patch_ground_elevations_m=zero_tuple,
        patch_agl_m=zero_tuple,
        patch_scale_factors=scale_tuple,
        center_gsd_cm_per_px=0.0,
    )


def load_terrain_context(
    resized_observation_shape: Tuple[int, int],
    config: SimulationConfig,
) -> TerrainContext:
    if not config.dem_path.exists():
        raise FileNotFoundError("DEM could not be found: %s" % config.dem_path)
    observation_grid_georef_path = (
        config.observation_grid_georef_path or config.observation_georef_path
    )
    if not observation_grid_georef_path.exists():
        raise FileNotFoundError(
            "Observation grid georeference raster could not be found: %s"
            % observation_grid_georef_path
        )

    observation_dataset = rio.open(str(observation_grid_georef_path))
    dem_dataset = rio.open(str(config.dem_path))
    observation_to_dem_transformer = Transformer.from_crs(
        observation_dataset.crs,
        dem_dataset.crs,
        always_xy=True,
    )
    return TerrainContext(
        dem_dataset=dem_dataset,
        observation_dataset=observation_dataset,
        observation_to_dem_transformer=observation_to_dem_transformer,
        resized_observation_shape=resized_observation_shape,
    )


def close_terrain_context(terrain_context: Optional[TerrainContext]) -> None:
    if terrain_context is None:
        return
    try:
        terrain_context.observation_dataset.close()
    except Exception:
        pass
    try:
        terrain_context.dem_dataset.close()
    except Exception:
        pass


def sample_ground_elevation_at_resized_pixel(
    pixel_x: float,
    pixel_y: float,
    terrain_context: TerrainContext,
) -> float:
    resized_height, resized_width = terrain_context.resized_observation_shape
    source_col = float(pixel_x) * (
        terrain_context.observation_dataset.width / float(resized_width)
    )
    source_row = float(pixel_y) * (
        terrain_context.observation_dataset.height / float(resized_height)
    )

    source_col = min(
        max(source_col, 0.0),
        max(0.0, terrain_context.observation_dataset.width - 1.0),
    )
    source_row = min(
        max(source_row, 0.0),
        max(0.0, terrain_context.observation_dataset.height - 1.0),
    )

    world_x, world_y = terrain_context.observation_dataset.transform * (
        source_col + 0.5,
        source_row + 0.5,
    )
    dem_x, dem_y = terrain_context.observation_to_dem_transformer.transform(
        world_x,
        world_y,
    )

    dem_bounds = terrain_context.dem_dataset.bounds
    if not (
        dem_bounds.left <= dem_x <= dem_bounds.right
        and dem_bounds.bottom <= dem_y <= dem_bounds.top
    ):
        raise ValueError(
            "Point is outside DEM bounds: x=%.3f y=%.3f" % (dem_x, dem_y)
        )

    sample = next(terrain_context.dem_dataset.sample([(dem_x, dem_y)]))[0]
    if not np.isfinite(sample):
        raise ValueError(
            "DEM sample is invalid at x=%.3f y=%.3f" % (dem_x, dem_y)
        )
    return float(sample)


def compute_altitude_simulation_state(
    observation_boxes: List[Tuple[int, int, int, int]],
    altitude_agl_m: float,
    terrain_context: TerrainContext,
    config: SimulationConfig,
) -> AltitudeSimulationState:
    patch_ground_elevations = []
    for box in observation_boxes:
        center_x = box[0] + (box[2] / 2.0)
        center_y = box[1] + (box[3] / 2.0)
        patch_ground_elevations.append(
            sample_ground_elevation_at_resized_pixel(
                center_x,
                center_y,
                terrain_context,
            )
        )

    center_ground_elevation_m = patch_ground_elevations[1]
    altitude_agl_m = clamp_altitude_agl(altitude_agl_m, config)
    altitude_msl_m = center_ground_elevation_m + altitude_agl_m
    patch_agl_values = tuple(
        max(config.minimum_patch_agl_m, altitude_msl_m - elevation_m)
        for elevation_m in patch_ground_elevations
    )
    patch_scale_factors = tuple(
        compute_scale_factor_for_altitude(patch_agl_m, config)
        for patch_agl_m in patch_agl_values
    )
    return AltitudeSimulationState(
        altitude_agl_m=altitude_agl_m,
        altitude_msl_m=float(altitude_msl_m),
        center_ground_elevation_m=float(center_ground_elevation_m),
        patch_ground_elevations_m=tuple(float(value) for value in patch_ground_elevations),
        patch_agl_m=tuple(float(value) for value in patch_agl_values),
        patch_scale_factors=tuple(float(value) for value in patch_scale_factors),
        center_gsd_cm_per_px=float(
            compute_virtual_camera_gsd_cm_per_px(altitude_agl_m, config)
        ),
    )


def _draw_alpha_panel(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    x0 = max(0, min(int(x0), image.shape[1] - 1))
    x1 = max(0, min(int(x1), image.shape[1] - 1))
    y0 = max(0, min(int(y0), image.shape[0] - 1))
    y1 = max(0, min(int(y1), image.shape[0] - 1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x1 - x0, y1 - y0), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _draw_rounded_rect(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    color: Tuple[int, int, int],
    thickness: int = -1,
) -> None:
    radius = max(0, min(int(radius), (x1 - x0) // 2, (y1 - y0) // 2))
    if radius < 2:
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
        return
    cv2.rectangle(image, (x0 + radius, y0), (x1 - radius, y1), color, thickness)
    cv2.rectangle(image, (x0, y0 + radius), (x1, y1 - radius), color, thickness)
    cv2.circle(image, (x0 + radius, y0 + radius), radius, color, thickness)
    cv2.circle(image, (x1 - radius, y0 + radius), radius, color, thickness)
    cv2.circle(image, (x0 + radius, y1 - radius), radius, color, thickness)
    cv2.circle(image, (x1 - radius, y1 - radius), radius, color, thickness)


def _draw_alpha_rounded_panel(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    x0 = max(0, min(int(x0), image.shape[1] - 1))
    x1 = max(0, min(int(x1), image.shape[1] - 1))
    y0 = max(0, min(int(y0), image.shape[0] - 1))
    y1 = max(0, min(int(y1), image.shape[0] - 1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    overlay = roi.copy()
    _draw_rounded_rect(overlay, 0, 0, x1 - x0, y1 - y0, radius, color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _draw_toggle_switch(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    is_on: bool,
    view_scale: float,
) -> None:
    radius = height // 2
    background = UI_COLORS["toggle_on"] if is_on else UI_COLORS["toggle_off"]
    cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), background, -1)
    cv2.circle(image, (x + radius, y + radius), radius, background, -1)
    cv2.circle(image, (x + width - radius, y + radius), radius, background, -1)

    knob_pad = max(3, int(round(4 * view_scale)))
    knob_x = x + width - radius - knob_pad if is_on else x + radius + knob_pad
    cv2.circle(image, (knob_x, y + radius), radius - knob_pad, UI_COLORS["toggle_knob"], -1)

    border = max(1, int(round(1.5 * view_scale)))
    cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), (200, 200, 210), border)
    cv2.circle(image, (x + radius, y + radius), radius, (200, 200, 210), border)
    cv2.circle(image, (x + width - radius, y + radius), radius, (200, 200, 210), border)


def _draw_text_with_shadow(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(
        image,
        text,
        (position[0] + 2, position[1] + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        UI_COLORS["text_shadow"],
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_info_panel(
    image: np.ndarray,
    lines: List[str],
    top_left: Tuple[int, int],
    font_scale: float,
    thickness: int,
    alpha: float = 0.55,
    padding: int = 18,
    corner_radius: int = 18,
) -> None:
    if not lines:
        return

    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for line in lines]
    max_width = max(width for width, _ in sizes)
    max_height = max(height for _, height in sizes)
    line_gap = int(max_height * 1.6)
    x, y = top_left
    panel_width = max_width + (2 * padding)
    panel_height = (line_gap * len(lines)) + padding
    panel_x0 = max(0, x - padding)
    panel_y0 = max(0, y - max_height - padding)
    panel_x1 = min(image.shape[1] - 1, panel_x0 + panel_width)
    panel_y1 = min(image.shape[0] - 1, panel_y0 + panel_height)

    _draw_alpha_rounded_panel(
        image,
        panel_x0,
        panel_y0,
        panel_x1,
        panel_y1,
        corner_radius,
        UI_COLORS["panel_bg"],
        alpha,
    )
    _draw_rounded_rect(
        image,
        panel_x0,
        panel_y0,
        panel_x1,
        panel_y1,
        corner_radius,
        UI_COLORS["panel_border"],
        max(1, thickness // 3),
    )

    for index, line in enumerate(lines):
        _draw_text_with_shadow(
            image,
            line,
            (x, y + (index * line_gap)),
            font_scale,
            UI_COLORS["text_primary"],
            thickness,
        )


def _build_runtime_buttons() -> List[dict]:
    return [
        {"key": "_panel_collapsed", "label": "", "hotkey": "H", "rect": (0, 0, 0, 0), "is_collapse": True},
        {"key": "info_panel", "label": "Bilgi", "hotkey": "B", "rect": (0, 0, 0, 0)},
        {"key": "trajectory", "label": "Trajektori", "hotkey": "T", "rect": (0, 0, 0, 0)},
        {"key": "roi_frame", "label": "ROI Cerceve", "hotkey": "O", "rect": (0, 0, 0, 0)},
        {"key": "tm_boxes", "label": "TM Kutular", "hotkey": "R", "rect": (0, 0, 0, 0)},
        {"key": "heading_arrow", "label": "Yon Oku", "hotkey": "Y", "rect": (0, 0, 0, 0)},
        {"key": "observation_boxes", "label": "Gozlem Kutulari", "hotkey": "G", "rect": (0, 0, 0, 0)},
    ]


def _draw_runtime_buttons(
    image: np.ndarray,
    ui_state: dict,
    buttons: List[dict],
    config: SimulationConfig,
) -> None:
    if image is None or not buttons:
        return

    view_scale = max(0.25, float(config.ui_button_scale))
    margin = max(12, int(round(18 * view_scale)))
    gap = max(6, int(round(10 * view_scale)))
    button_width = max(180, int(round(280 * view_scale)))
    button_height = max(38, int(round(56 * view_scale)))
    panel_padding = max(8, int(round(14 * view_scale)))
    header_height = max(28, int(round(38 * view_scale)))
    corner_radius = max(6, int(round(12 * view_scale)))
    button_radius = max(4, int(round(8 * view_scale)))
    font_scale = max(0.52, float(config.ui_button_font_scale) * view_scale)
    font_thickness = max(1, int(round(config.ui_button_thickness * view_scale)))
    toggle_width = max(46, int(round(72 * view_scale)))
    toggle_height = max(20, int(round(30 * view_scale)))

    x0 = margin
    y0 = margin
    hover_key = ui_state.get("_hover_key")
    is_collapsed = bool(ui_state.get("_panel_collapsed", False))

    collapse_button = None
    content_buttons = []
    for button in buttons:
        if button.get("is_collapse"):
            collapse_button = button
        else:
            content_buttons.append(button)

    if is_collapsed:
        size = max(34, int(round(44 * view_scale)))
        x1 = x0 + size
        y1 = y0 + size
        fill = UI_COLORS["collapse_hover"] if hover_key == "_panel_collapsed" else UI_COLORS["collapse_btn"]
        _draw_alpha_rounded_panel(image, x0, y0, x1, y1, button_radius, UI_COLORS["panel_bg"], 0.65)
        _draw_rounded_rect(image, x0, y0, x1, y1, button_radius, fill, -1)
        _draw_rounded_rect(image, x0, y0, x1, y1, button_radius, UI_COLORS["panel_border"], 1)
        if collapse_button is not None:
            collapse_button["rect"] = (x0, y0, size, size)
        bar_width = int(size * 0.45)
        bar_height = max(2, int(round(3 * view_scale)))
        bar_x = x0 + ((size - bar_width) // 2)
        bar_gap = max(4, int(round(6 * view_scale)))
        bar_center = y0 + (size // 2)
        for delta in (-bar_gap, 0, bar_gap):
            y_bar = bar_center + delta - (bar_height // 2)
            cv2.rectangle(image, (bar_x, y_bar), (bar_x + bar_width, y_bar + bar_height), UI_COLORS["text_primary"], -1)
        return

    panel_width = button_width + (2 * panel_padding)
    panel_height = (
        header_height
        + (2 * panel_padding)
        + (len(content_buttons) * button_height)
        + (max(0, len(content_buttons) - 1) * gap)
    )
    px0 = x0 - panel_padding
    py0 = y0 - panel_padding
    px1 = px0 + panel_width
    py1 = py0 + panel_height

    _draw_alpha_rounded_panel(image, px0, py0, px1, py1, corner_radius, UI_COLORS["panel_bg"], 0.60)
    _draw_rounded_rect(image, px0, py0, px1, py1, corner_radius, UI_COLORS["panel_border"], 1)
    _draw_alpha_rounded_panel(image, px0, py0, px1, py0 + header_height + panel_padding, corner_radius, UI_COLORS["header_bg"], 0.35)
    _draw_text_with_shadow(
        image,
        "GORUNUM",
        (x0 + int(round(8 * view_scale)), y0 + int(round(header_height * 0.62))),
        max(0.52, font_scale * 0.92),
        UI_COLORS["accent"],
        font_thickness,
    )

    if collapse_button is not None:
        size = max(26, int(round(34 * view_scale)))
        cb_x0 = px1 - size - max(4, int(round(6 * view_scale)))
        cb_y0 = py0 + max(4, int(round(6 * view_scale)))
        cb_x1 = cb_x0 + size
        cb_y1 = cb_y0 + size
        fill = UI_COLORS["collapse_hover"] if hover_key == "_panel_collapsed" else UI_COLORS["collapse_btn"]
        _draw_rounded_rect(image, cb_x0, cb_y0, cb_x1, cb_y1, button_radius // 2, fill, -1)
        _draw_rounded_rect(image, cb_x0, cb_y0, cb_x1, cb_y1, button_radius // 2, UI_COLORS["panel_border"], 1)
        collapse_button["rect"] = (cb_x0, cb_y0, size, size)
        line_width = int(size * 0.5)
        line_height = max(2, int(round(3 * view_scale)))
        line_x = cb_x0 + ((size - line_width) // 2)
        line_y = cb_y0 + (size // 2) - (line_height // 2)
        cv2.rectangle(image, (line_x, line_y), (line_x + line_width, line_y + line_height), UI_COLORS["text_primary"], -1)

    current_y = y0 + header_height
    for button in content_buttons:
        key = button["key"]
        is_on = bool(ui_state.get(key, False))
        is_hovered = hover_key == key
        fill = UI_COLORS["btn_hover_on"] if is_on and is_hovered else UI_COLORS["btn_on"] if is_on else UI_COLORS["btn_hover_off"] if is_hovered else UI_COLORS["btn_off"]
        edge = UI_COLORS["accent"] if is_hovered else UI_COLORS["panel_border"]
        button["rect"] = (x0, current_y, button_width, button_height)
        _draw_rounded_rect(image, x0, current_y, x0 + button_width, current_y + button_height, button_radius, fill, -1)
        _draw_rounded_rect(image, x0, current_y, x0 + button_width, current_y + button_height, button_radius, edge, 1)
        label = "%s [%s]" % (button["label"], button["hotkey"])
        _draw_text_with_shadow(
            image,
            label,
            (x0 + max(10, int(round(14 * view_scale))), current_y + int(round(button_height * 0.63))),
            font_scale,
            UI_COLORS["text_primary"],
            font_thickness,
        )
        toggle_x = x0 + button_width - toggle_width - max(8, int(round(10 * view_scale)))
        toggle_y = current_y + ((button_height - toggle_height) // 2)
        _draw_toggle_switch(image, toggle_x, toggle_y, toggle_width, toggle_height, is_on, view_scale)
        current_y += button_height + gap


def _runtime_buttons_mouse_cb(event: int, x: int, y: int, flags: int, userdata: dict) -> None:
    _ = flags
    if not isinstance(userdata, dict):
        return
    ui_state = userdata.get("state")
    buttons = userdata.get("buttons")
    if not isinstance(ui_state, dict) or not isinstance(buttons, list):
        return

    if event == cv2.EVENT_MOUSEMOVE:
        previous_hover = ui_state.get("_hover_key")
        ui_state["_hover_key"] = None
        for button in buttons:
            bx, by, bw, bh = button.get("rect", (0, 0, 0, 0))
            if bx <= x <= (bx + bw) and by <= y <= (by + bh):
                ui_state["_hover_key"] = button.get("key")
                break
        if previous_hover != ui_state.get("_hover_key"):
            ui_state["_dirty"] = True
        return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for button in buttons:
        bx, by, bw, bh = button.get("rect", (0, 0, 0, 0))
        if bx <= x <= (bx + bw) and by <= y <= (by + bh):
            key = button.get("key")
            ui_state[key] = not bool(ui_state.get(key, False))
            ui_state["_dirty"] = True
            break


def create_runtime_ui_state(config: SimulationConfig) -> dict:
    return {
        "info_panel": bool(config.show_info_panel),
        "trajectory": bool(config.show_trajectory),
        "roi_frame": bool(config.show_roi_frame),
        "tm_boxes": bool(config.show_tm_boxes),
        "heading_arrow": bool(config.show_heading_arrow),
        "observation_boxes": bool(config.show_observation_boxes),
        "_panel_collapsed": True,
        "_hover_key": None,
        "_dirty": True,
    }


def apply_runtime_ui_hotkey(key: int, ui_state: dict) -> bool:
    hotkey_map = {
        ord("b"): "info_panel",
        ord("B"): "info_panel",
        ord("t"): "trajectory",
        ord("T"): "trajectory",
        ord("o"): "roi_frame",
        ord("O"): "roi_frame",
        ord("r"): "tm_boxes",
        ord("R"): "tm_boxes",
        ord("y"): "heading_arrow",
        ord("Y"): "heading_arrow",
        ord("g"): "observation_boxes",
        ord("G"): "observation_boxes",
        ord("h"): "_panel_collapsed",
        ord("H"): "_panel_collapsed",
    }
    target = hotkey_map.get(key)
    if target is None:
        return False
    ui_state[target] = not bool(ui_state.get(target, False))
    ui_state["_dirty"] = True
    return True


def load_assets(config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, object]:
    validate_config(config)
    if is_georaster_path(config.reference_map_path):
        reference_map = load_grayscale_raster(config.reference_map_path)
    else:
        reference_map = load_grayscale_image(config.reference_map_path)

    if (
        config.align_observation_to_reference_grid
        and is_georaster_path(config.reference_map_path)
        and is_georaster_path(config.observation_map_path)
    ):
        observation_map = load_observation_aligned_to_reference_grid(
            config.observation_map_path,
            config.reference_map_path,
        )
    elif is_georaster_path(config.observation_map_path):
        observation_map = load_grayscale_raster(config.observation_map_path)
    else:
        observation_map = load_grayscale_image(config.observation_map_path)

    if observation_map.shape != reference_map.shape:
        observation_map = cv2.resize(
            observation_map,
            (reference_map.shape[1], reference_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    model = load_model_compat(config.model_path)
    return reference_map, observation_map, model


def clamp_observation_cursor(
    row: int,
    col: int,
    image_shape: Tuple[int, int],
    config: SimulationConfig,
) -> Tuple[int, int]:
    minimum, maximum_row, maximum_col = get_observation_cursor_limits(image_shape, config)
    clamped_row = min(max(row, minimum), maximum_row)
    clamped_col = min(max(col, minimum), maximum_col)
    return clamped_row, clamped_col


def sample_center_biased_coordinate(
    minimum: int,
    maximum: int,
    band_ratio: float,
) -> int:
    if maximum <= minimum:
        return int(minimum)

    center = (minimum + maximum) / 2.0
    half_span = (maximum - minimum) * max(0.05, min(1.0, float(band_ratio))) / 2.0
    band_min = max(minimum, int(round(center - half_span)))
    band_max = min(maximum, int(round(center + half_span)))
    if band_max <= band_min:
        return int(round(center))
    return random.randint(band_min, band_max)


def get_observation_boxes(
    row: int,
    col: int,
    heading_degrees: float,
    config: SimulationConfig,
) -> List[Tuple[int, int, int, int]]:
    _ = heading_degrees
    size = config.sample_window_size
    offset = config.template_offset
    return [
        (col - size - offset, row - size - offset, size, size),
        (col - size, row - size, size, size),
        (col - size + offset, row - size + offset, size, size),
    ]


def get_template_boxes_from_observation_boxes(
    observation_boxes: List[Tuple[int, int, int, int]],
    config: SimulationConfig,
) -> List[Tuple[int, int, int, int]]:
    inset = max(0, (config.sample_window_size - config.template_size) // 2)
    return [
        (x + inset, y + inset, config.template_size, config.template_size)
        for x, y, _, _ in observation_boxes
    ]


def rotate_square_capture(
    image: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        angle_degrees,
        1.0,
    )
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def extract_rotated_observation_window(
    observation_map: np.ndarray,
    observation_box: Tuple[int, int, int, int],
    heading_degrees: float,
    scale_factor: float,
    config: SimulationConfig,
) -> np.ndarray:
    capture_size = get_rotated_capture_size(config)
    center_x = observation_box[0] + (observation_box[2] // 2)
    center_y = observation_box[1] + (observation_box[3] // 2)
    capture_left = int(round(center_x - (capture_size / 2.0)))
    capture_top = int(round(center_y - (capture_size / 2.0)))
    capture_right = capture_left + capture_size
    capture_bottom = capture_top + capture_size

    raw_capture = observation_map[capture_top:capture_bottom, capture_left:capture_right]
    if raw_capture.shape[:2] != (capture_size, capture_size):
        pad_bottom = max(0, capture_size - raw_capture.shape[0])
        pad_right = max(0, capture_size - raw_capture.shape[1])
        raw_capture = cv2.copyMakeBorder(
            raw_capture,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_REPLICATE,
        )

    simulated_flight_capture = rotate_square_capture(raw_capture, heading_degrees)
    north_aligned_capture = rotate_square_capture(
        simulated_flight_capture,
        -heading_degrees,
    )

    scaled_window_size = get_scaled_observation_window_size(scale_factor, config)
    crop_start = (capture_size - scaled_window_size) // 2
    crop_end = crop_start + scaled_window_size
    scaled_window = north_aligned_capture[crop_start:crop_end, crop_start:crop_end]

    if scaled_window.shape[:2] != (scaled_window_size, scaled_window_size):
        pad_bottom = max(0, scaled_window_size - scaled_window.shape[0])
        pad_right = max(0, scaled_window_size - scaled_window.shape[1])
        scaled_window = cv2.copyMakeBorder(
            scaled_window,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_REPLICATE,
        )

    if scaled_window_size != config.sample_window_size:
        scaled_window = cv2.resize(
            scaled_window,
            (config.sample_window_size, config.sample_window_size),
            interpolation=cv2.INTER_NEAREST,
        )

    return scaled_window


def prepare_triplet_for_model(
    observation_windows: List[np.ndarray],
    config: SimulationConfig,
) -> np.ndarray:
    prepared_windows = []
    for observation_window in observation_windows:
        resized_window = cv2.resize(
            observation_window,
            (config.model_input_size, config.model_input_size),
            interpolation=cv2.INTER_NEAREST,
        )
        equalized_window = cv2.equalizeHist(resized_window)
        normalized_window = (equalized_window.astype(np.float32) - 127.5) / 127.5
        prepared_windows.append(normalized_window)

    return np.stack(prepared_windows, axis=0).reshape(
        -1,
        config.model_input_size,
        config.model_input_size,
        1,
    )


def predict_template_triplet(
    model: object,
    model_input_triplet: np.ndarray,
    config: SimulationConfig,
) -> List[np.ndarray]:
    predictions = model.predict(model_input_triplet, verbose=0)
    templates = []
    for prediction in predictions:
        prediction_2d = prediction.reshape(config.model_input_size, config.model_input_size)
        cropped_prediction = prediction_2d[
            config.crop_margin : config.model_input_size - config.crop_margin,
            config.crop_margin : config.model_input_size - config.crop_margin,
        ]
        scaled_prediction = np.clip(cropped_prediction, 0.0, 1.0)
        template_image = np.asarray(
            np.round(scaled_prediction * 255.0),
            dtype=np.uint8,
        )
        templates.append(template_image)
    return templates


def extract_template_triplet(
    observation_map: np.ndarray,
    row: int,
    col: int,
    heading_degrees: float,
    altitude_agl_m: float,
    terrain_context: Optional[TerrainContext],
    model: object,
    config: SimulationConfig,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[Tuple[int, int, int, int]],
    List[Tuple[int, int, int, int]],
    int,
    int,
    AltitudeSimulationState,
]:
    row, col = clamp_observation_cursor(row, col, observation_map.shape, config)
    observation_boxes = get_observation_boxes(row, col, heading_degrees, config)
    if is_altitude_scenario(config):
        if terrain_context is None:
            raise ValueError("Irtifa senaryosu icin terrain_context gerekli.")
        altitude_state = compute_altitude_simulation_state(
            observation_boxes,
            altitude_agl_m,
            terrain_context,
            config,
        )
    else:
        altitude_state = build_normal_altitude_state(len(observation_boxes))

    observation_windows = []
    for box, scale_factor in zip(
        observation_boxes,
        altitude_state.patch_scale_factors,
    ):
        observation_windows.append(
            extract_rotated_observation_window(
                observation_map,
                box,
                heading_degrees,
                scale_factor,
                config,
            )
        )

    model_input_triplet = prepare_triplet_for_model(observation_windows, config)
    templates = predict_template_triplet(model, model_input_triplet, config)
    template_boxes = get_template_boxes_from_observation_boxes(observation_boxes, config)

    return (
        templates,
        observation_windows,
        observation_boxes,
        template_boxes,
        row,
        col,
        altitude_state,
    )


def is_sqdiff_method(match_method: int) -> bool:
    return match_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)


def extract_match_score_and_location(
    response_map: np.ndarray,
    match_method: int,
) -> Tuple[float, Tuple[int, int]]:
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response_map)
    if is_sqdiff_method(match_method):
        return float(min_val), min_loc
    return float(max_val), max_loc


def run_template_match(
    reference_map: np.ndarray,
    template: np.ndarray,
    match_method: int,
) -> Tuple[float, Tuple[int, int]]:
    response_map = cv2.matchTemplate(reference_map, template, match_method, None)
    return extract_match_score_and_location(response_map, match_method)


def run_template_match_pyramid(
    search_region: np.ndarray,
    template: np.ndarray,
    config: SimulationConfig,
) -> Tuple[float, Tuple[int, int]]:
    match_method = config.match_method
    region_height, region_width = search_region.shape[:2]
    template_height, template_width = template.shape[:2]
    result_height = region_height - template_height + 1
    result_width = region_width - template_width + 1

    if result_height <= 0 or result_width <= 0:
        fallback_score = float("inf") if is_sqdiff_method(match_method) else float("-inf")
        return fallback_score, (0, 0)

    scale = float(config.coarse_scale)
    small_width = max(1, int(region_width * scale))
    small_height = max(1, int(region_height * scale))
    small_template_width = max(1, int(template_width * scale))
    small_template_height = max(1, int(template_height * scale))

    if (
        small_template_width > small_width
        or small_template_height > small_height
        or scale <= 0.0
    ):
        coarse_x = result_width // 2
        coarse_y = result_height // 2
    else:
        region_small = cv2.resize(
            search_region,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )
        template_small = cv2.resize(
            template,
            (small_template_width, small_template_height),
            interpolation=cv2.INTER_AREA,
        )
        coarse_response = cv2.matchTemplate(region_small, template_small, match_method, None)
        if coarse_response.size == 0:
            coarse_x = result_width // 2
            coarse_y = result_height // 2
        else:
            _, coarse_loc = extract_match_score_and_location(coarse_response, match_method)
            coarse_x = int(coarse_loc[0] / scale)
            coarse_y = int(coarse_loc[1] / scale)

    pad = max(8, int(max(template_width, template_height) * config.roi_pad_factor))
    x1 = max(0, coarse_x - pad)
    y1 = max(0, coarse_y - pad)
    x2 = min(result_width - 1, coarse_x + pad)
    y2 = min(result_height - 1, coarse_y + pad)

    if x2 < x1:
        x1 = x2 = max(0, min(coarse_x, result_width - 1))
    if y2 < y1:
        y1 = y2 = max(0, min(coarse_y, result_height - 1))

    region_roi = search_region[y1 : y2 + template_height, x1 : x2 + template_width]
    roi_response = cv2.matchTemplate(region_roi, template, match_method, None)
    score, top_left = extract_match_score_and_location(roi_response, match_method)
    return score, (top_left[0] + x1, top_left[1] + y1)


def match_three(
    search_region: np.ndarray,
    templates: List[np.ndarray],
    config: SimulationConfig,
) -> Tuple[List[Tuple[float, Tuple[int, int]]], str]:
    if config.use_pyramid_matching:
        worker = lambda template: run_template_match_pyramid(search_region, template, config)
        backend_label = "parallel-pyramid" if config.use_parallel_matching else "serial-pyramid"
    else:
        worker = lambda template: run_template_match(search_region, template, config.match_method)
        backend_label = "parallel-direct" if config.use_parallel_matching else "serial-direct"

    if config.use_parallel_matching:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, template) for template in templates]
            results = [future.result() for future in futures]
    else:
        results = [worker(template) for template in templates]

    return results, backend_label


def intersect_boxes(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    x = max(box_a[0], box_b[0])
    y = max(box_a[1], box_b[1])
    width = min(box_a[0] + box_a[2], box_b[0] + box_b[2]) - x
    height = min(box_a[1] + box_a[3], box_b[1] + box_b[3]) - y

    if width <= 0 or height <= 0:
        return None
    return (x, y, width, height)


def compute_intersection_box(
    boxes: List[Tuple[int, int, int, int]],
) -> Tuple[Tuple[int, int, int, int], str]:
    intersection_ab = intersect_boxes(boxes[0], boxes[1])
    intersection_bc = intersect_boxes(boxes[1], boxes[2])
    intersection_ac = intersect_boxes(boxes[0], boxes[2])

    if intersection_ab and intersection_bc:
        intersection_abc = intersect_boxes(intersection_ab, intersection_bc)
        if intersection_abc:
            return intersection_abc, "abc"

    if intersection_ab:
        return intersection_ab, "ab"
    if intersection_bc:
        return intersection_bc, "bc"
    if intersection_ac:
        return intersection_ac, "ac"

    return boxes[1], "center_fallback"


def get_search_window_box(
    reference_map_shape: Tuple[int, int],
    center: Tuple[int, int],
    window_size: int,
) -> Tuple[int, int, int, int]:
    height, width = reference_map_shape
    half_window = max(1, int(window_size // 2))
    center_x, center_y = center

    left = max(0, center_x - half_window)
    top = max(0, center_y - half_window)
    right = min(width, center_x + half_window)
    bottom = min(height, center_y + half_window)

    return left, top, right, bottom


def extract_search_region(
    reference_map: np.ndarray,
    previous_predicted_center: Optional[Tuple[int, int]],
    search_window_size: int,
    step_count: int,
    config: SimulationConfig,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int], str]:
    force_global = (
        previous_predicted_center is None
        or config.global_refresh_interval > 0
        and step_count > 0
        and (step_count % config.global_refresh_interval) == 0
    )
    if force_global:
        full_box = (0, 0, reference_map.shape[1], reference_map.shape[0])
        return reference_map, (0, 0), full_box, "global"

    search_box = get_search_window_box(
        reference_map.shape,
        previous_predicted_center,
        search_window_size,
    )
    left, top, right, bottom = search_box
    return reference_map[top:bottom, left:right], (left, top), search_box, "adaptive-roi"


def localize_template_triplet(
    search_region: np.ndarray,
    search_origin: Tuple[int, int],
    templates: List[np.ndarray],
    config: SimulationConfig,
) -> Tuple[List[float], List[Tuple[int, int, int, int]], Tuple[int, int, int, int], str, str]:
    scores = []
    matched_boxes = []

    match_results, match_backend = match_three(search_region, templates, config)

    for template, match_result in zip(templates, match_results):
        score, local_top_left = match_result
        scores.append(score)
        matched_boxes.append(
            (
                local_top_left[0] + search_origin[0],
                local_top_left[1] + search_origin[1],
                template.shape[1],
                template.shape[0],
            )
        )

    intersection_box, intersection_mode = compute_intersection_box(matched_boxes)
    return scores, matched_boxes, intersection_box, intersection_mode, match_backend


def get_box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return (box[0] + (box[2] // 2), box[1] + (box[3] // 2))


def compute_error_pixels(
    predicted_center: Tuple[int, int],
    actual_center: Tuple[int, int],
) -> float:
    return float(
        np.hypot(
            predicted_center[0] - actual_center[0],
            predicted_center[1] - actual_center[1],
        )
    )


def is_strict_triplet_alignment(
    matched_boxes: List[Tuple[int, int, int, int]],
    intersection_mode: str,
    config: SimulationConfig,
) -> bool:
    if intersection_mode != "abc" or len(matched_boxes) != 3:
        return False

    center_a = get_box_center(matched_boxes[0])
    center_b = get_box_center(matched_boxes[1])
    center_c = get_box_center(matched_boxes[2])

    delta_ab_x = float(center_b[0] - center_a[0])
    delta_ab_y = float(center_b[1] - center_a[1])
    delta_bc_x = float(center_c[0] - center_b[0])
    delta_bc_y = float(center_c[1] - center_b[1])
    midpoint_x = (center_a[0] + center_c[0]) / 2.0
    midpoint_y = (center_a[1] + center_c[1]) / 2.0
    midpoint_error = math.hypot(center_b[0] - midpoint_x, center_b[1] - midpoint_y)
    expected_offset = float(config.template_offset)
    step_error = max(
        abs(delta_ab_x - expected_offset),
        abs(delta_ab_y - expected_offset),
        abs(delta_bc_x - expected_offset),
        abs(delta_bc_y - expected_offset),
    )
    symmetry_error = max(
        abs(delta_ab_x - delta_bc_x),
        abs(delta_ab_y - delta_bc_y),
    )
    monotonic_diagonal = (
        delta_ab_x > 0.0
        and delta_ab_y > 0.0
        and delta_bc_x > 0.0
        and delta_bc_y > 0.0
    )
    alignment_error = max(midpoint_error, step_error, symmetry_error)
    return monotonic_diagonal and (
        alignment_error <= float(config.triplet_alignment_tolerance_px)
    )


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def calculate_fit_size(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
) -> Tuple[int, int]:
    scale = min(float(target_width) / image_width, float(target_height) / image_height)
    return (
        max(1, int(round(image_width * scale))),
        max(1, int(round(image_height * scale))),
    )


def resize_to_fit(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    if image_height == 0 or image_width == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    resized_width, resized_height = calculate_fit_size(
        image_width,
        image_height,
        target_width,
        target_height,
    )
    scale = min(float(target_width) / image_width, float(target_height) / image_height)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    return cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)


def get_dashboard_layout(
    config: SimulationConfig,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    dashboard_width, dashboard_height = config.display_size
    left_panel_width = int(dashboard_width * float(config.left_panel_width_ratio))
    map_width = (
        dashboard_width
        - (3 * config.panel_padding)
        - config.panel_gap
        - left_panel_width
    )
    stacked_panel_height = (
        dashboard_height
        - (2 * config.panel_padding)
        - config.panel_gap
    ) // 2

    observation_rect = (
        config.panel_padding,
        config.panel_padding,
        left_panel_width,
        stacked_panel_height,
    )
    template_rect = (
        config.panel_padding,
        config.panel_padding + stacked_panel_height + config.panel_gap,
        left_panel_width,
        stacked_panel_height,
    )
    map_rect = (
        config.panel_padding + left_panel_width + config.panel_gap,
        config.panel_padding,
        map_width,
        dashboard_height - (2 * config.panel_padding),
    )
    return observation_rect, template_rect, map_rect


def get_panel_content_rect(
    panel_rect: Tuple[int, int, int, int],
    config: SimulationConfig,
) -> Tuple[int, int, int, int]:
    x, y, width, height = panel_rect
    return (
        x + config.panel_inner_padding,
        y + config.panel_title_height + config.panel_inner_padding,
        width - (2 * config.panel_inner_padding),
        height - config.panel_title_height - (2 * config.panel_inner_padding),
    )


def draw_panel_frame(
    canvas: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    title: str,
    config: SimulationConfig,
) -> None:
    x, y, width, height = panel_rect
    cv2.rectangle(
        canvas,
        (x, y),
        (x + width, y + height),
        config.panel_background_color,
        -1,
    )
    cv2.rectangle(
        canvas,
        (x, y),
        (x + width, y + height),
        config.panel_border_color,
        2,
    )
    cv2.putText(
        canvas,
        title,
        (x + 12, y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        config.panel_title_color,
        2,
    )


def draw_panel(
    canvas: np.ndarray,
    image: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    title: str,
    config: SimulationConfig,
) -> None:
    draw_panel_frame(canvas, panel_rect, title, config)
    content_x, content_y, content_width, content_height = get_panel_content_rect(
        panel_rect,
        config,
    )
    fitted = resize_to_fit(ensure_bgr(image), content_width, content_height)
    fitted_height, fitted_width = fitted.shape[:2]
    paste_x = content_x + (content_width - fitted_width) // 2
    paste_y = content_y + (content_height - fitted_height) // 2
    canvas[paste_y : paste_y + fitted_height, paste_x : paste_x + fitted_width] = fitted


def get_reference_viewport_box(
    reference_map_shape: Tuple[int, int],
    predicted_intersection_box: Tuple[int, int, int, int],
    actual_intersection_box: Tuple[int, int, int, int],
    search_window_box: Tuple[int, int, int, int],
    search_mode: str,
    config: SimulationConfig,
) -> Tuple[int, int, int, int]:
    map_height, map_width = reference_map_shape
    if search_mode != "global":
        padding = int(config.reference_viewport_search_padding)
        search_width = int(search_window_box[2] - search_window_box[0])
        search_height = int(search_window_box[3] - search_window_box[1])
        viewport_width = min(
            map_width,
            max(
                int(config.reference_viewport_search_min_size),
                search_width + (2 * padding),
            ),
        )
        viewport_height = min(
            map_height,
            max(
                int(config.reference_viewport_search_min_size),
                search_height + (2 * padding),
            ),
        )
        center_x = (search_window_box[0] + search_window_box[2]) / 2.0
        center_y = (search_window_box[1] + search_window_box[3]) / 2.0
        viewport_left = int(round(center_x - (viewport_width / 2.0)))
        viewport_top = int(round(center_y - (viewport_height / 2.0)))
        viewport_left = min(max(viewport_left, 0), max(0, map_width - viewport_width))
        viewport_top = min(max(viewport_top, 0), max(0, map_height - viewport_height))
        viewport_right = viewport_left + viewport_width
        viewport_bottom = viewport_top + viewport_height
        return viewport_left, viewport_top, viewport_right, viewport_bottom

    relevant_boxes = [
        predicted_intersection_box,
        actual_intersection_box,
    ]
    left = min(box[0] for box in relevant_boxes)
    top = min(box[1] for box in relevant_boxes)
    right = max(box[0] + box[2] for box in relevant_boxes)
    bottom = max(box[1] + box[3] for box in relevant_boxes)
    padding = int(config.reference_viewport_padding)

    viewport_width = min(
        map_width,
        max(
            int(config.reference_viewport_base_size),
            int((right - left) + (2 * padding)),
        ),
    )
    viewport_height = min(
        map_height,
        max(
            int(config.reference_viewport_base_size),
            int((bottom - top) + (2 * padding)),
        ),
    )

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    viewport_left = int(round(center_x - (viewport_width / 2.0)))
    viewport_top = int(round(center_y - (viewport_height / 2.0)))
    viewport_left = min(max(viewport_left, 0), max(0, map_width - viewport_width))
    viewport_top = min(max(viewport_top, 0), max(0, map_height - viewport_height))
    viewport_right = viewport_left + viewport_width
    viewport_bottom = viewport_top + viewport_height
    return viewport_left, viewport_top, viewport_right, viewport_bottom


def create_reference_preview_state(
    reference_map: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    viewport_box: Tuple[int, int, int, int],
    config: SimulationConfig,
) -> ReferencePreviewState:
    content_x, content_y, content_width, content_height = get_panel_content_rect(
        panel_rect,
        config,
    )
    viewport_left, viewport_top, viewport_right, viewport_bottom = viewport_box
    reference_crop = reference_map[viewport_top:viewport_bottom, viewport_left:viewport_right]
    if reference_crop.size == 0:
        reference_crop = reference_map
        viewport_left = 0
        viewport_top = 0
        viewport_right = reference_map.shape[1]
        viewport_bottom = reference_map.shape[0]

    preview_width, preview_height = calculate_fit_size(
        reference_crop.shape[1],
        reference_crop.shape[0],
        content_width,
        content_height,
    )
    preview_image = cv2.resize(
        reference_crop,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )
    preview_image = ensure_bgr(preview_image)

    paste_x = content_x + (content_width - preview_width) // 2
    paste_y = content_y + (content_height - preview_height) // 2

    return ReferencePreviewState(
        panel_rect=panel_rect,
        paste_x=paste_x,
        paste_y=paste_y,
        preview_width=preview_width,
        preview_height=preview_height,
        scale_x=preview_width / float(viewport_right - viewport_left),
        scale_y=preview_height / float(viewport_bottom - viewport_top),
        viewport_left=viewport_left,
        viewport_top=viewport_top,
        viewport_width=viewport_right - viewport_left,
        viewport_height=viewport_bottom - viewport_top,
        base_preview=preview_image,
    )


def scale_point_to_preview(
    point: Tuple[int, int],
    preview_state: ReferencePreviewState,
) -> Tuple[int, int]:
    return (
        int(round((point[0] - preview_state.viewport_left) * preview_state.scale_x)),
        int(round((point[1] - preview_state.viewport_top) * preview_state.scale_y)),
    )


def scale_box_to_preview(
    box: Tuple[int, int, int, int],
    preview_state: ReferencePreviewState,
) -> Tuple[int, int, int, int]:
    left = int(round((box[0] - preview_state.viewport_left) * preview_state.scale_x))
    top = int(round((box[1] - preview_state.viewport_top) * preview_state.scale_y))
    right = int(
        round(
            ((box[0] + box[2]) - preview_state.viewport_left) * preview_state.scale_x
        )
    )
    bottom = int(
        round(
            ((box[1] + box[3]) - preview_state.viewport_top) * preview_state.scale_y
        )
    )
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return left, top, right, bottom


def draw_scaled_path(
    preview_image: np.ndarray,
    points: List[Tuple[int, int]],
    preview_state: ReferencePreviewState,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return

    margin = max(50, int(round(120 / max(preview_state.scale_x, preview_state.scale_y, 1e-6))))
    filtered_points = [
        point
        for point in points
        if (
            (preview_state.viewport_left - margin)
            <= point[0]
            <= (preview_state.viewport_left + preview_state.viewport_width + margin)
            and (preview_state.viewport_top - margin)
            <= point[1]
            <= (preview_state.viewport_top + preview_state.viewport_height + margin)
        )
    ]
    if len(filtered_points) < 2:
        return

    scaled_points = [
        scale_point_to_preview(point, preview_state) for point in filtered_points
    ]
    cv2.polylines(
        preview_image,
        [np.array(scaled_points, dtype=np.int32)],
        False,
        color,
        thickness,
    )


def draw_heading_arrow(
    image: np.ndarray,
    origin: Tuple[int, int],
    heading_degrees: float,
    length: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if length <= 0:
        return

    direction_x, direction_y = get_heading_vector(heading_degrees)
    arrow_tip = (
        int(round(origin[0] + (direction_x * length))),
        int(round(origin[1] + (direction_y * length))),
    )
    cv2.arrowedLine(
        image,
        origin,
        arrow_tip,
        color,
        thickness,
        cv2.LINE_AA,
        0,
        0.24,
    )


def create_observation_context_view(
    observation_map: np.ndarray,
    observation_boxes: List[Tuple[int, int, int, int]],
    actual_boxes: List[Tuple[int, int, int, int]],
    actual_intersection_box: Tuple[int, int, int, int],
    heading_degrees: float,
    ui_state: dict,
    config: SimulationConfig,
) -> np.ndarray:
    context_boxes = observation_boxes + actual_boxes + [actual_intersection_box]
    left = max(0, min(box[0] for box in context_boxes) - config.observation_context_margin)
    top = max(0, min(box[1] for box in context_boxes) - config.observation_context_margin)
    right = min(
        observation_map.shape[1],
        max(box[0] + box[2] for box in context_boxes) + config.observation_context_margin,
    )
    bottom = min(
        observation_map.shape[0],
        max(box[1] + box[3] for box in context_boxes) + config.observation_context_margin,
    )

    view = ensure_bgr(observation_map[top:bottom, left:right])
    if view.size == 0:
        return np.zeros((config.sample_window_size, config.sample_window_size, 3), dtype=np.uint8)

    if ui_state.get("observation_boxes", True):
        for index, box in enumerate(observation_boxes):
            box_left = box[0] - left
            box_top = box[1] - top
            box_right = box_left + box[2]
            box_bottom = box_top + box[3]
            cv2.rectangle(
                view,
                (box_left, box_top),
                (box_right, box_bottom),
                TEMPLATE_COLORS[index],
                2,
            )
            cv2.putText(
                view,
                "O%d" % (index + 1),
                (box_left + 8, box_top + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                TEMPLATE_COLORS[index],
                2,
            )

        for index, box in enumerate(actual_boxes):
            box_left = box[0] - left
            box_top = box[1] - top
            box_right = box_left + box[2]
            box_bottom = box_top + box[3]
            cv2.rectangle(
                view,
                (box_left, box_top),
                (box_right, box_bottom),
                TEMPLATE_COLORS[index],
                1,
            )

    intersection_left = actual_intersection_box[0] - left
    intersection_top = actual_intersection_box[1] - top
    intersection_right = intersection_left + actual_intersection_box[2]
    intersection_bottom = intersection_top + actual_intersection_box[3]
    cv2.rectangle(
        view,
        (intersection_left, intersection_top),
        (intersection_right, intersection_bottom),
        config.actual_intersection_color,
        2,
    )

    actual_center = get_box_center(actual_intersection_box)
    local_center = (actual_center[0] - left, actual_center[1] - top)
    cv2.circle(view, local_center, 6, config.actual_intersection_color, -1)
    if ui_state.get("heading_arrow", True):
        draw_heading_arrow(
            view,
            local_center,
            heading_degrees,
            max(36, min(view.shape[0], view.shape[1]) // 5),
            config.heading_indicator_color,
            2,
        )
    cv2.putText(
        view,
        "Baslik: %s" % format_heading_label(heading_degrees),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        config.panel_title_color,
        2,
    )
    return view


def _draw_observation_tile(
    canvas: np.ndarray,
    image: np.ndarray,
    top_left: Tuple[int, int],
    size: int,
    label: str,
    color: Tuple[int, int, int],
    subtitle: str,
) -> None:
    x, y = top_left
    tile = cv2.resize(
        ensure_bgr(image),
        (size, size),
        interpolation=cv2.INTER_AREA if image.shape[0] >= size else cv2.INTER_NEAREST,
    )
    canvas[y : y + size, x : x + size] = tile
    cv2.rectangle(canvas, (x, y), (x + size, y + size), color, 2)
    cv2.putText(
        canvas,
        label,
        (x + 10, y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        color,
        2,
    )
    cv2.putText(
        canvas,
        subtitle,
        (x + 10, y + size - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
    )


def format_patch_subtitle(
    patch_index: int,
    altitude_state: AltitudeSimulationState,
    config: SimulationConfig,
) -> str:
    if is_altitude_scenario(config):
        return "%.1fm | x%.2f" % (
            altitude_state.patch_agl_m[patch_index],
            altitude_state.patch_scale_factors[patch_index],
        )
    return "normal | x%.2f" % altitude_state.patch_scale_factors[patch_index]


def create_observation_view(
    observation_map: np.ndarray,
    observation_boxes: List[Tuple[int, int, int, int]],
    actual_boxes: List[Tuple[int, int, int, int]],
    actual_intersection_box: Tuple[int, int, int, int],
    observation_windows: List[np.ndarray],
    altitude_state: AltitudeSimulationState,
    heading_degrees: float,
    ui_state: dict,
    config: SimulationConfig,
) -> np.ndarray:
    _ = (
        actual_boxes,
        actual_intersection_box,
        observation_windows,
        heading_degrees,
        ui_state,
    )
    if len(observation_boxes) < 3:
        return np.zeros(
            (config.sample_window_size, config.sample_window_size, 3),
            dtype=np.uint8,
        )

    hero_size = config.sample_window_size
    padding = 16
    title_height = 34
    canvas_width = hero_size
    canvas_height = (padding * 2) + title_height + hero_size
    canvas = np.full(
        (canvas_height, canvas_width, 3),
        config.panel_background_color,
        dtype=np.uint8,
    )

    center_x = (canvas_width - hero_size) // 2
    top_y = padding
    cv2.putText(
        canvas,
        "Mavi Pencere - Ham Raster Crop",
        (padding, top_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        config.panel_title_color,
        2,
    )
    hero_y = top_y + title_height
    raw_blue_window = extract_padded_patch(observation_map, observation_boxes[2])
    _draw_observation_tile(
        canvas,
        raw_blue_window,
        (center_x, hero_y),
        hero_size,
        "O3",
        TEMPLATE_COLORS[2],
        "raw 544x544",
    )

    return canvas


def create_template_strip(
    templates: List[np.ndarray],
    config: SimulationConfig,
) -> np.ndarray:
    if len(templates) < 3:
        return np.zeros(
            (config.sample_window_size, config.sample_window_size, 3),
            dtype=np.uint8,
        )

    hero_size = config.sample_window_size
    padding = 16
    title_height = 34
    strip_width = hero_size
    strip_height = (padding * 2) + title_height + hero_size

    strip = np.full(
        (strip_height, strip_width, 3),
        config.panel_background_color,
        dtype=np.uint8,
    )

    blue_template = cv2.resize(
        ensure_bgr(templates[2]),
        (hero_size, hero_size),
        interpolation=cv2.INTER_AREA,
    )
    tile_x = (strip_width - hero_size) // 2
    tile_y = padding + title_height
    strip[tile_y : tile_y + hero_size, tile_x : tile_x + hero_size] = blue_template
    cv2.rectangle(
        strip,
        (tile_x, tile_y),
        (tile_x + hero_size, tile_y + hero_size),
        TEMPLATE_COLORS[2],
        2,
    )
    cv2.putText(
        strip,
        "Mavi Pencere - Model Ciktisi",
        (padding, padding + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        config.panel_title_color,
        2,
    )

    return strip


def extract_padded_patch(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    x, y, width, height = box
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image.shape[1], x + width)
    y2 = min(image.shape[0], y + height)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((height, width), dtype=image.dtype)

    pad_top = max(0, -y)
    pad_left = max(0, -x)
    pad_bottom = max(0, (y + height) - image.shape[0])
    pad_right = max(0, (x + width) - image.shape[1])
    if pad_top or pad_bottom or pad_left or pad_right:
        patch = cv2.copyMakeBorder(
            patch,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE,
        )
    return patch


def compose_triplet_diagnostic_image(
    reference_map: np.ndarray,
    actual_boxes: List[Tuple[int, int, int, int]],
    matched_boxes: List[Tuple[int, int, int, int]],
    templates: List[np.ndarray],
    observation_windows: List[np.ndarray],
    score_values: List[float],
    error_pixels: float,
    point_label: str,
    altitude_state: AltitudeSimulationState,
    config: SimulationConfig,
) -> np.ndarray:
    tile_size = int(config.diagnostic_tile_size)
    padding = 18
    gap = 14
    header_height = 116
    row_gap = 18
    cols = 4
    rows = 3
    canvas_width = (padding * 2) + (cols * tile_size) + ((cols - 1) * gap)
    canvas_height = (
        header_height
        + (padding * 2)
        + (rows * tile_size)
        + ((rows - 1) * row_gap)
    )
    canvas = np.full(
        (canvas_height, canvas_width, 3),
        config.panel_background_color,
        dtype=np.uint8,
    )

    header_lines = [
        "%s | scenario=%s | err=%.1f px"
        % (
            point_label,
            get_scenario_label(config),
            error_pixels,
        ),
        "Kolonlar: Gozlem | Gercek Patch | Model Template | Eslesen Patch",
    ]
    if is_altitude_scenario(config):
        header_lines[0] += " | alt=%.1f m agl | gsd=%.2f cm/px" % (
            altitude_state.altitude_agl_m,
            altitude_state.center_gsd_cm_per_px,
        )
    for line_index, line in enumerate(header_lines):
        cv2.putText(
            canvas,
            line,
            (padding, 32 + (line_index * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.74,
            config.panel_title_color,
            2,
            cv2.LINE_AA,
        )

    column_titles = (
        "Gozlem",
        "Gercek Patch",
        "Model Template",
        "Eslesen Patch",
    )
    for column_index, title in enumerate(column_titles):
        title_x = padding + (column_index * (tile_size + gap))
        cv2.putText(
            canvas,
            title,
            (title_x, header_height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            config.panel_title_color,
            2,
            cv2.LINE_AA,
        )

    for row_index in range(rows):
        base_y = header_height + padding + (row_index * (tile_size + row_gap))
        row_label = "O%d %s | score=%.4f" % (
            row_index + 1,
            format_patch_subtitle(row_index, altitude_state, config),
            score_values[row_index],
        )
        cv2.putText(
            canvas,
            row_label,
            (padding, base_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            TEMPLATE_COLORS[row_index],
            2,
            cv2.LINE_AA,
        )
        row_images = [
            observation_windows[row_index],
            extract_padded_patch(reference_map, actual_boxes[row_index]),
            templates[row_index],
            extract_padded_patch(reference_map, matched_boxes[row_index]),
        ]
        for column_index, row_image in enumerate(row_images):
            tile_x = padding + (column_index * (tile_size + gap))
            tile = cv2.resize(
                ensure_bgr(row_image),
                (tile_size, tile_size),
                interpolation=cv2.INTER_NEAREST,
            )
            canvas[base_y : base_y + tile_size, tile_x : tile_x + tile_size] = tile
            border_color = (
                TEMPLATE_COLORS[row_index]
                if column_index in (0, 1, 2, 3)
                else config.panel_border_color
            )
            cv2.rectangle(
                canvas,
                (tile_x, base_y),
                (tile_x + tile_size, base_y + tile_size),
                border_color,
                2,
            )

    return canvas


def run_template_diagnostics(
    reference_map: np.ndarray,
    observation_map: np.ndarray,
    model: object,
    terrain_context: Optional[TerrainContext],
    config: SimulationConfig,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.diagnostic_output_dir / ("template_diag_" + timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    heading_degrees = normalize_heading_degrees(config.initial_heading_degrees)

    for case_index, (row_seed, col_seed) in enumerate(
        config.diagnostic_benchmark_points,
        start=1,
    ):
        row, col = clamp_observation_cursor(
            row_seed,
            col_seed,
            observation_map.shape,
            config,
        )
        (
            templates,
            observation_windows,
            observation_boxes,
            actual_boxes,
            row,
            col,
            altitude_state,
        ) = extract_template_triplet(
            observation_map,
            row,
            col,
            heading_degrees,
            config.initial_altitude_agl_m,
            terrain_context,
            model,
            config,
        )
        actual_intersection_box, _ = compute_intersection_box(actual_boxes)
        actual_center = get_box_center(actual_intersection_box)
        search_region, search_origin, search_window_box, search_mode = extract_search_region(
            reference_map,
            actual_center,
            config.base_search_window_size,
            0,
            config,
        )
        (
            score_values,
            matched_boxes,
            predicted_intersection_box,
            intersection_mode,
            match_backend,
        ) = localize_template_triplet(search_region, search_origin, templates, config)
        predicted_center = get_box_center(predicted_intersection_box)
        error_pixels = compute_error_pixels(predicted_center, actual_center)

        case_name = "case_%02d_r%d_c%d" % (case_index, row, col)
        diagnostic_image = compose_triplet_diagnostic_image(
            reference_map,
            actual_boxes,
            matched_boxes,
            templates,
            observation_windows,
            score_values,
            error_pixels,
            case_name,
            altitude_state,
            config,
        )
        cv2.imwrite(str(output_dir / (case_name + "_triptych.png")), diagnostic_image)

        metadata = {
            "case_index": case_index,
            "row": row,
            "col": col,
            "actual_center": list(actual_center),
            "predicted_center": list(predicted_center),
            "error_pixels": float(error_pixels),
            "score_values": [float(value) for value in score_values],
            "intersection_mode": intersection_mode,
            "search_mode": search_mode,
            "match_backend": match_backend,
            "search_window_box": list(search_window_box),
            "actual_boxes": [list(box) for box in actual_boxes],
            "matched_boxes": [list(box) for box in matched_boxes],
            "patch_scale_factors": [
                float(value) for value in altitude_state.patch_scale_factors
            ],
            "patch_agl_m": [float(value) for value in altitude_state.patch_agl_m],
        }
        (output_dir / (case_name + "_meta.json")).write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        summary_rows.append(metadata)

    summary = {
        "scenario_mode": get_scenario_label(config),
        "reference_map_path": str(config.reference_map_path),
        "observation_map_path": str(config.observation_map_path),
        "model_path": str(config.model_path),
        "case_count": len(summary_rows),
        "cases": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("Template diagnostics exported to %s" % output_dir)
    return output_dir


def draw_hud(
    canvas: np.ndarray,
    map_rect: Tuple[int, int, int, int],
    score_values: List[float],
    observation_cursor: Tuple[int, int],
    predicted_center: Tuple[int, int],
    actual_center: Tuple[int, int],
    error_pixels: float,
    step_count: int,
    last_action: str,
    heading_degrees: float,
    altitude_state: AltitudeSimulationState,
    intersection_mode: str,
    search_mode: str,
    match_backend: str,
    search_window_size: int,
    ui_state: dict,
    config: SimulationConfig,
) -> None:
    x, y, width, height = map_rect
    if ui_state.get("info_panel", True):
        hud_lines = [
            "SCN: %s" % get_scenario_label(config),
            "HDG: %s" % format_heading_label(heading_degrees),
            "ERR: %.1f px" % error_pixels,
            "ROI: %d px" % search_window_size,
            "CMD: %s" % get_action_label(last_action),
        ]
        if is_altitude_scenario(config):
            hud_lines.insert(2, "ALT: %.1f m AGL" % altitude_state.altitude_agl_m)
            hud_lines.insert(3, "GSD: %.2f cm/px" % altitude_state.center_gsd_cm_per_px)
        draw_info_panel(
            canvas,
            hud_lines,
            top_left=(x + 28, y + 96),
            font_scale=1.00,
            thickness=2,
            alpha=0.55,
            padding=18,
            corner_radius=18,
        )

    help_line_1 = "WASD hareket | Q/E donus"
    if is_altitude_scenario(config):
        help_line_1 += " | +/- irtifa"
    help_line_2 = "H panel | B bilgi | T iz | O ROI | R TM | Y yon | G gozlem | ESC/X cikis"
    cv2.putText(
        canvas,
        help_line_1,
        (x + 12, y + height - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        config.panel_title_color,
        2,
    )
    cv2.putText(
        canvas,
        help_line_2,
        (x + 12, y + height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        config.panel_title_color,
        2,
    )


def draw_localization_dashboard(
    observation_rect: Tuple[int, int, int, int],
    template_rect: Tuple[int, int, int, int],
    reference_preview_state: ReferencePreviewState,
    observation_view: np.ndarray,
    template_strip: np.ndarray,
    matched_boxes: List[Tuple[int, int, int, int]],
    predicted_intersection_box: Tuple[int, int, int, int],
    actual_intersection_box: Tuple[int, int, int, int],
    search_window_box: Tuple[int, int, int, int],
    predicted_history: List[Tuple[int, int]],
    actual_history: List[Tuple[int, int]],
    score_values: List[float],
    observation_cursor: Tuple[int, int],
    step_count: int,
    last_action: str,
    heading_degrees: float,
    altitude_state: AltitudeSimulationState,
    intersection_mode: str,
    search_mode: str,
    match_backend: str,
    search_window_size: int,
    ui_state: dict,
    runtime_ui_buttons: List[dict],
    config: SimulationConfig,
) -> np.ndarray:
    dashboard_width, dashboard_height = config.display_size
    canvas = np.full(
        (dashboard_height, dashboard_width, 3),
        config.dashboard_background_color,
        dtype=np.uint8,
    )

    map_rect = reference_preview_state.panel_rect
    preview = reference_preview_state.base_preview.copy()

    predicted_center = get_box_center(predicted_intersection_box)
    actual_center = get_box_center(actual_intersection_box)

    overlay_thickness = max(2, int(round(6 * reference_preview_state.scale_x)))
    marker_radius = max(4, int(round(14 * reference_preview_state.scale_x)))

    if ui_state.get("trajectory", True):
        draw_scaled_path(
            preview,
            actual_history,
            reference_preview_state,
            config.actual_path_color,
            overlay_thickness,
        )
        draw_scaled_path(
            preview,
            predicted_history,
            reference_preview_state,
            config.predicted_path_color,
            overlay_thickness,
        )

    if ui_state.get("tm_boxes", True):
        for index, box in enumerate(matched_boxes):
            scaled_box = scale_box_to_preview(box, reference_preview_state)
            cv2.rectangle(
                preview,
                (scaled_box[0], scaled_box[1]),
                (scaled_box[2], scaled_box[3]),
                TEMPLATE_COLORS[index],
                overlay_thickness,
            )
            cv2.putText(
                preview,
                "T%d" % (index + 1),
                (scaled_box[0] + 6, scaled_box[1] + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                TEMPLATE_COLORS[index],
                2,
            )

    scaled_actual_box = scale_box_to_preview(actual_intersection_box, reference_preview_state)
    scaled_predicted_box = scale_box_to_preview(predicted_intersection_box, reference_preview_state)
    scaled_search_box = scale_box_to_preview(
        (
            search_window_box[0],
            search_window_box[1],
            search_window_box[2] - search_window_box[0],
            search_window_box[3] - search_window_box[1],
        ),
        reference_preview_state,
    )
    scaled_actual_center = scale_point_to_preview(actual_center, reference_preview_state)
    scaled_predicted_center = scale_point_to_preview(predicted_center, reference_preview_state)

    if ui_state.get("roi_frame", True):
        cv2.rectangle(
            preview,
            (scaled_search_box[0], scaled_search_box[1]),
            (scaled_search_box[2], scaled_search_box[3]),
            config.search_window_color,
            max(1, overlay_thickness - 1),
        )

    cv2.rectangle(
        preview,
        (scaled_actual_box[0], scaled_actual_box[1]),
        (scaled_actual_box[2], scaled_actual_box[3]),
        config.actual_intersection_color,
        overlay_thickness,
    )
    cv2.rectangle(
        preview,
        (scaled_predicted_box[0], scaled_predicted_box[1]),
        (scaled_predicted_box[2], scaled_predicted_box[3]),
        config.predicted_intersection_color,
        overlay_thickness,
    )
    cv2.circle(preview, scaled_actual_center, marker_radius, config.actual_intersection_color, -1)
    cv2.circle(
        preview,
        scaled_predicted_center,
        marker_radius,
        config.predicted_intersection_color,
        -1,
    )
    cv2.line(
        preview,
        scaled_actual_center,
        scaled_predicted_center,
        config.error_line_color,
        overlay_thickness,
    )
    if ui_state.get("heading_arrow", True):
        draw_heading_arrow(
            preview,
            scaled_actual_center,
            heading_degrees,
            max(marker_radius * 4, 28),
            config.heading_indicator_color,
            max(2, overlay_thickness - 1),
        )

    draw_panel(
        canvas,
        observation_view,
        observation_rect,
        config.observation_panel_title,
        config,
    )
    draw_panel(
        canvas,
        template_strip,
        template_rect,
        config.template_panel_title,
        config,
    )
    draw_panel_frame(canvas, map_rect, config.reference_panel_title, config)
    canvas[
        reference_preview_state.paste_y : reference_preview_state.paste_y
        + reference_preview_state.preview_height,
        reference_preview_state.paste_x : reference_preview_state.paste_x
        + reference_preview_state.preview_width,
    ] = preview

    draw_hud(
        canvas,
        map_rect,
        score_values,
        observation_cursor,
        predicted_center,
        actual_center,
        compute_error_pixels(predicted_center, actual_center),
        step_count,
        last_action,
        heading_degrees,
        altitude_state,
        intersection_mode,
        search_mode,
        match_backend,
        search_window_size,
        ui_state,
        config,
    )
    if config.ui_buttons_enabled:
        _draw_runtime_buttons(canvas, ui_state, runtime_ui_buttons, config)

    return canvas


def move_observation_cursor(
    row: int,
    col: int,
    action: str,
    image_shape: Tuple[int, int],
    heading_degrees: float,
    config: SimulationConfig,
) -> Tuple[int, int]:
    move_x = 0.0
    move_y = 0.0

    if action == "forward":
        move_x, move_y = rotate_image_offset(0.0, -float(config.step_size), heading_degrees)
    elif action == "backward":
        move_x, move_y = rotate_image_offset(0.0, float(config.step_size), heading_degrees)
    elif action == "strafe_right":
        move_x, move_y = rotate_image_offset(float(config.step_size), 0.0, heading_degrees)
    elif action == "strafe_left":
        move_x, move_y = rotate_image_offset(-float(config.step_size), 0.0, heading_degrees)

    row += int(round(move_y))
    col += int(round(move_x))

    return clamp_observation_cursor(row, col, image_shape, config)


def apply_control_action(
    row: int,
    col: int,
    heading_degrees: float,
    altitude_agl_m: float,
    action: str,
    image_shape: Tuple[int, int],
    config: SimulationConfig,
) -> Tuple[int, int, float, float]:
    if action == "rotate_left":
        return (
            row,
            col,
            normalize_heading_degrees(heading_degrees - config.rotation_step_degrees),
            altitude_agl_m,
        )
    if action == "rotate_right":
        return (
            row,
            col,
            normalize_heading_degrees(heading_degrees + config.rotation_step_degrees),
            altitude_agl_m,
        )
    if action == "altitude_up":
        if not is_altitude_scenario(config):
            return row, col, heading_degrees, altitude_agl_m
        return (
            row,
            col,
            heading_degrees,
            clamp_altitude_agl(altitude_agl_m + config.altitude_step_m, config),
        )
    if action == "altitude_down":
        if not is_altitude_scenario(config):
            return row, col, heading_degrees, altitude_agl_m
        return (
            row,
            col,
            heading_degrees,
            clamp_altitude_agl(altitude_agl_m - config.altitude_step_m, config),
        )

    row, col = move_observation_cursor(
        row,
        col,
        action,
        image_shape,
        heading_degrees,
        config,
    )
    return row, col, heading_degrees, altitude_agl_m


def get_action_from_key(key: int) -> str:
    if key in UP_KEYS:
        return "forward"
    if key in DOWN_KEYS:
        return "backward"
    if key in LEFT_KEYS:
        return "strafe_left"
    if key in RIGHT_KEYS:
        return "strafe_right"
    if key in ROTATE_LEFT_KEYS:
        return "rotate_left"
    if key in ROTATE_RIGHT_KEYS:
        return "rotate_right"
    if key in ALTITUDE_UP_KEYS:
        return "altitude_up"
    if key in ALTITUDE_DOWN_KEYS:
        return "altitude_down"
    return ""


def print_localization_status(
    score_values: List[float],
    matched_boxes: List[Tuple[int, int, int, int]],
    predicted_intersection_box: Tuple[int, int, int, int],
    actual_intersection_box: Tuple[int, int, int, int],
    row: int,
    col: int,
    error_pixels: float,
    step_count: int,
    last_action: str,
    heading_degrees: float,
    altitude_state: AltitudeSimulationState,
    intersection_mode: str,
    search_mode: str,
    match_backend: str,
    search_window_size: int,
    config: SimulationConfig,
) -> None:
    strict_triplet_lock = is_strict_triplet_alignment(
        matched_boxes,
        intersection_mode,
        config,
    )
    print("cursor=(row=%d, col=%d)" % (row, col))
    print(
        "scores=(%.4f, %.4f, %.4f) scenario=%s intersection_mode=%s search=%s backend=%s"
        % (
            score_values[0],
            score_values[1],
            score_values[2],
            get_scenario_label(config),
            intersection_mode,
            search_mode,
            match_backend,
        )
    )
    print("matched_boxes=%s" % (matched_boxes,))
    status_line = (
        "predicted_intersection=%s actual_intersection=%s error=%.1fpx step=%d action=%s heading=%s"
        % (
            predicted_intersection_box,
            actual_intersection_box,
            error_pixels,
            step_count,
            get_action_label(last_action),
            format_heading_label(heading_degrees),
        )
    )
    if is_altitude_scenario(config):
        status_line += " alt=%.1fm agl gsd=%.2fcm/px" % (
            altitude_state.altitude_agl_m,
            altitude_state.center_gsd_cm_per_px,
        )
    status_line += " window=%d lock=%s" % (
        search_window_size,
        "strict-triplet" if strict_triplet_lock else "partial",
    )
    print(status_line)


def update_search_window_size(
    current_search_window_size: int,
    matched_boxes: List[Tuple[int, int, int, int]],
    intersection_mode: str,
    config: SimulationConfig,
) -> int:
    if is_strict_triplet_alignment(matched_boxes, intersection_mode, config):
        return config.base_search_window_size
    if intersection_mode in ("abc", "ab", "bc", "ac"):
        return min(
            config.max_search_window_size,
            current_search_window_size + config.search_window_growth_step,
        )
    return min(
        config.max_search_window_size,
        current_search_window_size + config.search_window_failure_growth,
    )


def choose_initial_cursor(
    observation_map_shape: Tuple[int, int],
    config: SimulationConfig,
) -> Tuple[int, int]:
    if not config.random_start:
        return clamp_observation_cursor(
            config.initial_row,
            config.initial_col,
            observation_map_shape,
            config,
        )

    minimum, maximum_row, maximum_col = get_observation_cursor_limits(
        observation_map_shape,
        config,
    )
    return (
        sample_center_biased_coordinate(
            minimum,
            maximum_row,
            config.random_start_middle_band_ratio,
        ),
        sample_center_biased_coordinate(
            minimum,
            maximum_col,
            config.random_start_middle_band_ratio,
        ),
    )


def main() -> None:
    config = SimulationConfig()
    reference_map, observation_map, model = load_assets(config)
    terrain_context: Optional[TerrainContext] = None

    try:
        if is_altitude_scenario(config):
            terrain_context = load_terrain_context(observation_map.shape, config)

        if config.diagnostic_benchmark_enabled:
            run_template_diagnostics(
                reference_map,
                observation_map,
                model,
                terrain_context,
                config,
            )
            if config.diagnostic_benchmark_only:
                return

        observation_rect, template_rect, map_rect = get_dashboard_layout(config)

        row, col = choose_initial_cursor(observation_map.shape, config)
        predicted_history = []
        actual_history = []
        step_count = 0
        last_action = ""
        heading_degrees = normalize_heading_degrees(config.initial_heading_degrees)
        altitude_agl_m = clamp_altitude_agl(config.initial_altitude_agl_m, config)
        previous_predicted_center = None
        search_window_size = config.base_search_window_size
        runtime_ui_state = create_runtime_ui_state(config)
        runtime_ui_buttons = _build_runtime_buttons() if config.ui_buttons_enabled else []
        runtime_ui_context = {
            "state": runtime_ui_state,
            "buttons": runtime_ui_buttons,
        }

        cv2.namedWindow(config.dashboard_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            config.dashboard_window_name,
            config.display_size[0],
            config.display_size[1],
        )
        if config.ui_buttons_enabled:
            cv2.setMouseCallback(
                config.dashboard_window_name,
                _runtime_buttons_mouse_cb,
                runtime_ui_context,
            )

        while True:
            current_search_window_size = search_window_size
            (
                templates,
                observation_windows,
                observation_boxes,
                actual_boxes,
                row,
                col,
                altitude_state,
            ) = extract_template_triplet(
                observation_map,
                row,
                col,
                heading_degrees,
                altitude_agl_m,
                terrain_context,
                model,
                config,
            )
            altitude_agl_m = altitude_state.altitude_agl_m
            actual_intersection_box, _ = compute_intersection_box(actual_boxes)
            search_anchor_center = previous_predicted_center
            if search_anchor_center is None and step_count == 0:
                search_anchor_center = get_box_center(actual_intersection_box)
            search_region, search_origin, search_window_box, search_mode = (
                extract_search_region(
                    reference_map,
                    search_anchor_center,
                    current_search_window_size,
                    step_count,
                    config,
                )
            )
            score_values, matched_boxes, predicted_intersection_box, intersection_mode, match_backend = (
                localize_template_triplet(search_region, search_origin, templates, config)
            )

            predicted_center = get_box_center(predicted_intersection_box)
            actual_center = get_box_center(actual_intersection_box)
            error_pixels = compute_error_pixels(predicted_center, actual_center)
            strict_triplet_lock = is_strict_triplet_alignment(
                matched_boxes,
                intersection_mode,
                config,
            )
            if strict_triplet_lock:
                previous_predicted_center = predicted_center
            elif previous_predicted_center is None:
                previous_predicted_center = search_anchor_center
            search_window_size = update_search_window_size(
                current_search_window_size,
                matched_boxes,
                intersection_mode,
                config,
            )

            predicted_history.append(predicted_center)
            actual_history.append(actual_center)
            predicted_history = predicted_history[-config.path_history_limit :]
            actual_history = actual_history[-config.path_history_limit :]
            reference_viewport_box = get_reference_viewport_box(
                reference_map.shape,
                predicted_intersection_box,
                actual_intersection_box,
                search_window_box,
                search_mode,
                config,
            )
            reference_preview_state = create_reference_preview_state(
                reference_map,
                map_rect,
                reference_viewport_box,
                config,
            )

            observation_view = create_observation_view(
                observation_map,
                observation_boxes,
                actual_boxes,
                actual_intersection_box,
                observation_windows,
                altitude_state,
                heading_degrees,
                runtime_ui_state,
                config,
            )
            template_strip = create_template_strip(templates, config)

            print_localization_status(
                score_values,
                matched_boxes,
                predicted_intersection_box,
                actual_intersection_box,
                row,
                col,
                error_pixels,
                step_count,
                last_action,
                heading_degrees,
                altitude_state,
                intersection_mode,
                search_mode,
                match_backend,
                current_search_window_size,
                config,
            )

            dashboard = draw_localization_dashboard(
                observation_rect=observation_rect,
                template_rect=template_rect,
                reference_preview_state=reference_preview_state,
                observation_view=observation_view,
                template_strip=template_strip,
                matched_boxes=matched_boxes,
                predicted_intersection_box=predicted_intersection_box,
                actual_intersection_box=actual_intersection_box,
                search_window_box=search_window_box,
                predicted_history=predicted_history,
                actual_history=actual_history,
                score_values=score_values,
                observation_cursor=(row, col),
                step_count=step_count,
                last_action=last_action,
                heading_degrees=heading_degrees,
                altitude_state=altitude_state,
                intersection_mode=intersection_mode,
                search_mode=search_mode,
                match_backend=match_backend,
                search_window_size=current_search_window_size,
                ui_state=runtime_ui_state,
                runtime_ui_buttons=runtime_ui_buttons,
                config=config,
            )
            should_exit = False
            while True:
                cv2.imshow(config.dashboard_window_name, dashboard)
                runtime_ui_state["_dirty"] = False
                key = cv2.waitKeyEx(30)

                if runtime_ui_state.get("_dirty"):
                    dashboard = draw_localization_dashboard(
                        observation_rect=observation_rect,
                        template_rect=template_rect,
                        reference_preview_state=reference_preview_state,
                        observation_view=observation_view,
                        template_strip=template_strip,
                        matched_boxes=matched_boxes,
                        predicted_intersection_box=predicted_intersection_box,
                        actual_intersection_box=actual_intersection_box,
                        search_window_box=search_window_box,
                        predicted_history=predicted_history,
                        actual_history=actual_history,
                        score_values=score_values,
                        observation_cursor=(row, col),
                        step_count=step_count,
                        last_action=last_action,
                        heading_degrees=heading_degrees,
                        altitude_state=altitude_state,
                        intersection_mode=intersection_mode,
                        search_mode=search_mode,
                        match_backend=match_backend,
                        search_window_size=current_search_window_size,
                        ui_state=runtime_ui_state,
                        runtime_ui_buttons=runtime_ui_buttons,
                        config=config,
                    )
                    continue

                if key == -1:
                    continue

                if config.ui_buttons_enabled and apply_runtime_ui_hotkey(
                    key,
                    runtime_ui_state,
                ):
                    dashboard = draw_localization_dashboard(
                        observation_rect=observation_rect,
                        template_rect=template_rect,
                        reference_preview_state=reference_preview_state,
                        observation_view=observation_view,
                        template_strip=template_strip,
                        matched_boxes=matched_boxes,
                        predicted_intersection_box=predicted_intersection_box,
                        actual_intersection_box=actual_intersection_box,
                        search_window_box=search_window_box,
                        predicted_history=predicted_history,
                        actual_history=actual_history,
                        score_values=score_values,
                        observation_cursor=(row, col),
                        step_count=step_count,
                        last_action=last_action,
                        heading_degrees=heading_degrees,
                        altitude_state=altitude_state,
                        intersection_mode=intersection_mode,
                        search_mode=search_mode,
                        match_backend=match_backend,
                        search_window_size=current_search_window_size,
                        ui_state=runtime_ui_state,
                        runtime_ui_buttons=runtime_ui_buttons,
                        config=config,
                    )
                    continue

                if key in EXIT_KEYS:
                    should_exit = True
                    break

                action = get_action_from_key(key)
                if action:
                    row, col, heading_degrees, altitude_agl_m = apply_control_action(
                        row,
                        col,
                        heading_degrees,
                        altitude_agl_m,
                        action,
                        observation_map.shape,
                        config,
                    )
                    last_action = action
                    step_count += 1
                    break

                print("Unrecognized key code: %s" % key)

            if should_exit:
                break
    finally:
        close_terrain_context(terrain_context)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
