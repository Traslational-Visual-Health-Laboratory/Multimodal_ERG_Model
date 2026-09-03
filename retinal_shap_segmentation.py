# ─── Imports ──────────────────────────────────────────────────────────────────
import os
from PVBM.DiscSegmenter import DiscSegmenter
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
import tensorflow as tf
import shap
from PIL import Image
from scipy.ndimage import binary_fill_holes, center_of_mass
from scipy.stats import entropy as scipy_entropy
from skimage.filters import gaussian
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from skimage.morphology import (
    binary_closing, disk,
    remove_small_holes, remove_small_objects,
)

# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY — IMAGE EXTENSION RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

def resolve_image_path(path: str) -> str:
    """
    If `path` already exists as-is, returns it unchanged.
    If it doesn't exist, tries appending each extension in img_extensions and
    returns the first one that exists. Raises FileNotFoundError if none match.
    """
    if os.path.isfile(path):
        return path

    # Strip any extension the user may have typed (just in case)
    base, ext = os.path.splitext(path)
    root = base if ext.lower() in img_extensions else path

    for extension in img_extensions:
        candidate = root + extension
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"No image found for '{path}'. "
        f"Tried: {[root + e for e in img_extensions]}"
    )

def load_original_image(img_path: str) -> np.ndarray:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read the image: {img_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def load_and_preprocess_image(
    image_rgb: np.ndarray,
    target_size: tuple,
    backbone_name: str,
) -> np.ndarray:
    """Resizes and applies the architecture-specific preprocessing.
    Returns array (1, H, W, 3) ready for the model."""
    model_preprocessors = {
        "DenseNet201":        tf.keras.applications.densenet.preprocess_input,
        "MobileNetV2":        tf.keras.applications.mobilenet_v2.preprocess_input,
        "ResNet50":           tf.keras.applications.resnet50.preprocess_input,
        "InceptionV3":        tf.keras.applications.inception_v3.preprocess_input,
        "Xception":           tf.keras.applications.xception.preprocess_input,
        "VGG16":              tf.keras.applications.vgg16.preprocess_input,
        "VGG19":              tf.keras.applications.vgg19.preprocess_input,
        "ResNet101":          tf.keras.applications.resnet.preprocess_input,
        "InceptionResNetV21": tf.keras.applications.inception_resnet_v2.preprocess_input,
        "MobileNet":          tf.keras.applications.mobilenet.preprocess_input,
    }
    preprocessor = model_preprocessors.get(backbone_name)
    if preprocessor is None:
        raise ValueError(
            f"Model '{backbone_name}' not recognized. "
            f"Options: {list(model_preprocessors.keys())}"
        )
    img = cv2.resize(image_rgb, target_size).astype(np.float32)
    img = preprocessor(img)
    return np.expand_dims(img, axis=0)  # (1, H, W, 3)

def load_background_from_folder(
    background_dir: str,
    target_size: tuple,
    backbone_name: str,
    n_background: int,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> np.ndarray:
    """Loads up to n_background images, preprocesses them and returns (N, H, W, 3)."""
    paths = [
        os.path.join(background_dir, f)
        for f in os.listdir(background_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ]
    if not paths:
        raise FileNotFoundError(
            f"No images found in: {background_dir}"
        )
    np.random.shuffle(paths)
    paths = paths[:n_background]

    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [SHAP background] Could not read: {path}, skipping.")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pre = load_and_preprocess_image(img_rgb, target_size, backbone_name)
        images.append(img_pre[0])

    if not images:
        raise ValueError("Could not load any valid background image.")

    bg_array = np.array(images, dtype=np.float32)
    print(f"  [SHAP background] {bg_array.shape[0]} images loaded from '{background_dir}'")
    return bg_array

def deep_shap(model, image_tensor, background_images, pred_index: int) -> tuple:
    """SHAP values via GradientExplainer.

    Returns
    -------
    importance_map : np.ndarray (H, W)  — sum of |SHAP| across channels (unsigned).
    sign_map       : np.ndarray (H, W)  — signed sum of SHAP across channels.
                     Positive → pixel pushes towards the positive class (pred_index).
                     Negative → pixel pushes away from the positive class.
    """
    explainer   = shap.GradientExplainer(model, background_images)
    shap_values = explainer.shap_values(image_tensor)

    print(f"  [SHAP] type: {type(shap_values)}, ", end="")

    if isinstance(shap_values, list):
        print(f"len={len(shap_values)}, shape[{pred_index}]={np.array(shap_values[pred_index]).shape}")
        sv = np.array(shap_values[pred_index])
    else:
        sv = np.array(shap_values)
        print(f"shape={sv.shape}")

    # Collapse extra outputs dimension if present: (n, H, W, C, k) → (n, H, W, C)
    if sv.ndim == 5:
        idx = pred_index if sv.shape[-1] > pred_index else 0
        sv  = sv[..., idx]

    while sv.ndim < 4:
        sv = sv[np.newaxis]

    sv_sample = sv[0]  # (H, W, C)

    if sv_sample.ndim == 2:
        return np.abs(sv_sample).astype(np.float32), sv_sample.astype(np.float32)

    if sv_sample.ndim != 3:
        raise RuntimeError(
            f"Unexpected SHAP values shape: {sv_sample.shape} "
            f"(original shape: {sv.shape})"
        )

    importance_map = np.sum(np.abs(sv_sample), axis=-1).astype(np.float32)  # (H, W)
    sign_map       = np.sum(sv_sample,          axis=-1).astype(np.float32)  # (H, W)
    return importance_map, -sign_map

def normalize_map(input_map: np.ndarray) -> np.ndarray:
    """Normalizes a 2-D map to the [0, 1] range."""
    vmin, vmax = input_map.min(), input_map.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(input_map, dtype=np.float32)
    return ((input_map - vmin) / (vmax - vmin)).astype(np.float32)

def segment_retina(
    img_path: str,
    seg_path: str,
) -> dict:
    """
    Segments the image into 6 mutually exclusive regions and returns
    a dictionary with the boolean masks and key coordinates.

    Regions
    -------
    vessels             : blood vasculature
    optic_disc_mask     : optic disc (excluding vessels)
    faz_mask            : foveal avascular zone (excluding vessels and disc)
    macula_mask         : macular region (excluding vessels, disc and FAZ)
    parenchyma_mask     : retinal parenchyma (rest of the field)
    background_mask     : outside the eye
    """
    # deferred import to avoid a DLL error when importing the module

    img_path  = resolve_image_path(img_path)
    image     = Image.open(img_path)
    img_array = np.array(image)
    h, w      = img_array.shape[:2]

    # ── Vessels ──
    seg      = np.array(Image.open(seg_path)) / 255.0
    vessels  = remove_small_holes(
        remove_small_objects((seg > 0.5).astype(bool), min_size=64),
        area_threshold=64,
    )

    # ── Optic disc ──
    disc_segmentation = np.array(DiscSegmenter().segment(image_path=img_path)) > 0
    ys, xs            = np.where(disc_segmentation)
    cx, cy            = float(np.mean(xs)), float(np.mean(ys))
    disc_diameter = 2 * np.sqrt(len(xs) / np.pi)
    disc_radius   = disc_diameter / 2

    # ── Macula and FAZ ──
    direction     = 1 if cx < w / 2 else -1
    macula_x_est  = float(np.clip(cx + direction * 2.5 * disc_diameter, 0, w - 1))
    macula_y_est  = cy
    roi_radius    = int(disc_diameter * 0.8)

    y0, y1 = int(max(0, macula_y_est - roi_radius)), int(min(h, macula_y_est + roi_radius))
    x0, x1 = int(max(0, macula_x_est - roi_radius)), int(min(w, macula_x_est + roi_radius))

    roi = img_array[y0:y1, x0:x1, 1].copy().astype(float)
    roi[vessels[y0:y1, x0:x1]] = 255.0
    min_y, min_x       = np.unravel_index(np.argmin(gaussian(roi, sigma=5)), roi.shape)
    macula_x, macula_y = x0 + min_x, y0 + min_y

    macula_radius = disc_radius * 2.0
    faz_radius    = disc_radius * 0.4

    # ── Circular masks ──
    yy, xx = np.ogrid[:h, :w]

    optic_disc_mask_raw = (xx - cx)**2       + (yy - cy)**2       <= disc_radius**2
    macula_mask_raw     = (xx - macula_x)**2 + (yy - macula_y)**2 <= macula_radius**2
    faz_mask_raw         = (xx - macula_x)**2 + (yy - macula_y)**2 <= faz_radius**2

    # ── Retinal field (excludes black corners) ──
    brightness  = img_array.mean(axis=2)
    mask_bright = brightness > 15
    mask_bright = binary_closing(mask_bright, disk(20))

    labeled        = label(mask_bright)
    regions        = regionprops(labeled)
    largest_region = max(regions, key=lambda r: r.area)
    retina_mask    = binary_fill_holes(labeled == largest_region.label)

    # ── Strict hierarchy (no overlap) ──
    optic_disc_mask = optic_disc_mask_raw & ~vessels
    faz_mask        = faz_mask_raw        & ~vessels & ~optic_disc_mask_raw
    macula_mask     = macula_mask_raw     & ~vessels & ~optic_disc_mask_raw & ~faz_mask_raw
    parenchyma_mask = retina_mask         & ~vessels & ~optic_disc_mask_raw & ~macula_mask_raw
    background_mask = ~retina_mask

    # ── Verification ──
    all_masks = (vessels.astype(int) + optic_disc_mask.astype(int) + faz_mask.astype(int)
             + macula_mask.astype(int) + parenchyma_mask.astype(int) + background_mask.astype(int))
    assert all_masks.max() <= 1, "There are pixels assigned to more than one region!"

    return {
        "image":            image,
        "img_array":        img_array,
        "h": h, "w": w,
        "vessels":          vessels,
        "optic_disc_mask":  optic_disc_mask,
        "faz_mask":         faz_mask,
        "macula_mask":      macula_mask,
        "parenchyma_mask":  parenchyma_mask,
        "background_mask":  background_mask,
        # useful coordinates
        "cx": cx, "cy": cy,
        "macula_x": macula_x, "macula_y": macula_y,
        "disc_radius":   disc_radius,
        "macula_radius": macula_radius,
        "faz_radius":    faz_radius,
    }

def calculate_shap_metrics(
    norm_map: np.ndarray,
    seg: dict,
    sign_map: np.ndarray | None = None,
) -> dict:
    """
    Computes global and per-region metrics over the normalized [0,1] SHAP map.

    Global metrics
    --------------
    center_of_mass_x/y    : Centroid weighted by SHAP importance. Indicates which
                            spatial region the model is "looking at". If it falls in
                            the macula it validates that the model learned the
                            correct pathological region.
    entropy               : Shannon entropy of the SHAP map histogram.
                            High = diffuse importance; Low = concentrated importance.
    mean_importance       : Mean of the map. Low values = selective model.
    max_importance        : Maximum value of the map.
    std_importance        : Std. dev. of the map. High = strong contrast between zones.
    gini_coefficient      : 0 = uniform; 1 = concentrated in a single pixel.
    top5pct_area          : % of the area that concentrates the top 5% of importance.
    top5pct_mean_importance: Mean importance within that top 5%.

    Global sign metrics (if sign_map is available)
    -------------------------------------------------------
    signed_mean           : Mean of the signed SHAP (>0 = global positive red flag).
    signed_std            : Std. dev. of the signed map.
    pct_positive_pixels   : % of pixels with positive SHAP (push towards the positive class).
    pct_negative_pixels   : % of pixels with negative SHAP (push away from the positive class).
    net_signed_sum        : Total signed sum (indicates the model's net direction).

    Per-region metrics
    -------------------
    shap_sum              : Sum of SHAP in the region.
    shap_mean             : Mean SHAP (comparable across regions of different sizes).
    shap_std              : Intra-region std. dev. (heterogeneity).
    pct_of_total_shap     : % that the region contributes to the total SHAP sum.
    pct_of_image_area     : % of total pixels occupied by the region.
    shap_area_ratio       : pct_of_total_shap / pct_of_image_area.
                            >1 overrepresented, <1 underrepresented.
    n_pixels              : Number of pixels in the region.

    Per-region sign metrics (if sign_map is available)
    --------------------------------------------------------
    signed_mean           : Mean signed SHAP in the region.
                            >0 = region pushes the prediction towards the positive
                            class on average.
                            <0 = region pushes the prediction away from the positive
                            class on average.
    signed_sum            : Sum of the signed SHAP in the region.
    pct_positive_pixels   : % of positive pixels within the region.
    pct_negative_pixels   : % of negative pixels within the region.
    net_direction         : "positive", "negative" or "mixed" based on |signed_mean|
                            vs a threshold (0.05 * max_abs).
    """
    h, w = norm_map.shape

    # ── Global ──
    com_y, com_x = center_of_mass(norm_map)

    # ── Center of mass inside FAZ / Macula ──
    com_xi, com_yi = int(round(com_x)), int(round(com_y))
    com_xi = int(np.clip(com_xi, 0, w - 1))
    com_yi = int(np.clip(com_yi, 0, h - 1))

    com_in_faz    = bool(seg["faz_mask"][com_yi, com_xi])
    com_in_macula = bool(seg["macula_mask"][com_yi, com_xi]) or com_in_faz  # FAZ ⊂ Macula

    hist, _ = np.histogram(norm_map, bins=256, range=(0.0, 1.0), density=False)
    hist_p  = hist / (hist.sum() + 1e-12)
    entropy_val = float(scipy_entropy(hist_p + 1e-12))

    vals_flat = norm_map.flatten()
    mean_imp  = float(vals_flat.mean())
    max_imp   = float(vals_flat.max())
    std_imp   = float(vals_flat.std())

    # Gini coefficient
    sorted_vals = np.sort(vals_flat)
    n           = len(sorted_vals)
    cumsum      = np.cumsum(sorted_vals)
    gini        = float(1 - 2 * cumsum.sum() / (n * (cumsum[-1] + 1e-12)))

    # Top-5% area
    thresh_top5          = float(np.percentile(vals_flat, 95))
    mask_top5            = norm_map >= thresh_top5
    top5_pct_area        = float(mask_top5.sum()) / (h * w) * 100.0
    top5_mean_importance = float(norm_map[mask_top5].mean()) if mask_top5.any() else 0.0

    # ── Global sign metrics ──
    signed_global = {}
    if sign_map is not None:
        sv_flat = sign_map.flatten()
        signed_global = {
            "signed_mean":         round(float(sv_flat.mean()), 6),
            "signed_std":          round(float(sv_flat.std()),  6),
            "pct_positive_pixels": round(float((sv_flat > 0).sum()) / len(sv_flat) * 100.0, 4),
            "pct_negative_pixels": round(float((sv_flat < 0).sum()) / len(sv_flat) * 100.0, 4),
            "net_signed_sum":      round(float(sv_flat.sum()), 6),
        }

    # ── Per region ──
    regions = {
        "vessels":    seg["vessels"],
        "optic_disc": seg["optic_disc_mask"],
        "faz":        seg["faz_mask"],
        "macula":     seg["macula_mask"],
        "parenchyma": seg["parenchyma_mask"],
        "background": seg["background_mask"],
    }

    total_sum    = float(norm_map.sum()) + 1e-12
    total_pixels = h * w

    # "mixed" threshold for net_direction: 5% of the absolute range of the sign map
    max_abs_sign    = float(np.abs(sign_map).max()) if sign_map is not None else 1.0
    mixed_threshold = 0.05 * max_abs_sign

    shap_by_region = {}
    for region_name, mask in regions.items():
        if mask.shape != norm_map.shape:
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_resized = mask

        region_vals = norm_map[mask_resized]
        n_pixels    = int(mask_resized.sum())
        region_sum  = float(region_vals.sum())
        region_mean = float(region_vals.mean()) if n_pixels > 0 else 0.0
        region_std  = float(region_vals.std())  if n_pixels > 0 else 0.0

        pct_shap = region_sum / total_sum * 100.0
        pct_area = n_pixels / total_pixels * 100.0
        ratio    = pct_shap / pct_area if pct_area > 0 else 0.0

        entry = {
            "shap_sum":          round(region_sum, 6),
            "shap_mean":         round(region_mean, 6),
            "shap_std":          round(region_std, 6),
            "pct_of_total_shap": round(pct_shap, 4),
            "pct_of_image_area": round(pct_area, 4),
            "shap_area_ratio":   round(ratio, 4),
            "n_pixels":          n_pixels,
        }

        # ── Per-region sign metrics ──
        if sign_map is not None:
            region_sv = sign_map[mask_resized]
            region_signed_mean = float(region_sv.mean()) if n_pixels > 0 else 0.0
            region_signed_sum  = float(region_sv.sum())
            pct_pos = float((region_sv > 0).sum()) / n_pixels * 100.0 if n_pixels > 0 else 0.0
            pct_neg = float((region_sv < 0).sum()) / n_pixels * 100.0 if n_pixels > 0 else 0.0
            if   abs(region_signed_mean) < mixed_threshold: net_dir = "mixed"
            elif region_signed_mean > 0:                    net_dir = "positive"
            else:                                            net_dir = "negative"

            entry["signed_mean"]         = round(region_signed_mean, 6)
            entry["signed_sum"]          = round(region_signed_sum,  6)
            entry["pct_positive_pixels"] = round(pct_pos, 4)
            entry["pct_negative_pixels"] = round(pct_neg, 4)
            entry["net_direction"]       = net_dir

        shap_by_region[region_name] = entry

    result = {
        "center_of_mass_x":       round(float(com_x), 2),
        "center_of_mass_y":       round(float(com_y), 2),
        "com_in_faz":             com_in_faz,
        "com_in_macula":          com_in_macula,
        "entropy":                round(entropy_val, 6),
        "mean_importance":        round(mean_imp, 6),
        "max_importance":         round(max_imp, 6),
        "std_importance":         round(std_imp, 6),
        "gini_coefficient":       round(gini, 6),
        "top5pct_area":           round(top5_pct_area, 4),
        "top5pct_mean_importance":round(top5_mean_importance, 6),
        "shap_by_region":         shap_by_region,
    }
    result.update(signed_global)
    return result

def save_combined_figure(
    image_rgb:  np.ndarray,
    norm_map:   np.ndarray,
    sign_map:   np.ndarray | None,
    seg:        dict,
    metrics:    dict,
    img_name:   str,
    model_name: str,
    prediction: float,
    save_dir:   str,
    alpha_shap: float = 0.50,
    alpha_seg:  float = 0.55,
    cmap_shap:  str   = "jet",
) -> str:
    """Generates and saves the figure with 4 subplots + 2 colorbars without distortion.

    Layout:
        4 equally spaced subplots (Fundus | SHAP Importance | SHAP Sign | Segmentation).
        The 2 colorbars are positioned with fig.add_axes() using the real
        coordinates of ax1 and ax2 AFTER matplotlib has computed them,
        placing them right next to each subplot without invading the neighbor.
    """
    from matplotlib.lines import Line2D

    image = seg["image"]
    h, w  = seg["h"], seg["w"]

    # ── Dimensions ───────────────────────────────────────────────────────────
    col_w_in   = 5.0
    img_h_in   = col_w_in * (h / w)
    suptitle_h = 0.35
    title_h    = 0.55
    fig_w      = col_w_in * 4
    fig_h      = img_h_in + suptitle_h + title_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        1, 4,
        left   = 0.01,
        right  = 0.99,
        bottom = suptitle_h / fig_h,
        top    = 1.0 - title_h / fig_h,
        wspace = 0.27,
    )

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    extent = [0, w, h, 0]

    # ── Subplot 1: Original fundus ───────────────────────────────────────────
    ax0.imshow(image_rgb, extent=extent, aspect="equal")
    ax0.set_xlim(0, w); ax0.set_ylim(h, 0)
    ax0.set_title("Fundus Image", fontsize=18, fontweight="bold", pad=8)
    ax0.axis("off")

    # ── Shared mask: pixels without SHAP information ────────────────────────
    # norm_map < 1e-6 is used for both subplots, so the area without SHAP
    # remains completely transparent (does not paint dark blue over the disc)
    no_shap_mask = norm_map < 1e-6

    # ── Subplot 2: SHAP Importance ───────────────────────────────────────────
    ax1.imshow(image_rgb, extent=extent, aspect="equal")
    map_plot = np.ma.masked_where(no_shap_mask, norm_map)
    im_shap = ax1.imshow(
        map_plot, extent=extent, aspect="equal",
        cmap=cmap_shap, alpha=alpha_shap, vmin=0, vmax=1,
    )
    ax1.set_xlim(0, w); ax1.set_ylim(h, 0)
    ax1.set_title("SHAP Importance", fontsize=18, fontweight="bold", pad=8)
    ax1.axis("off")

    # ── Subplot 3: SHAP Sign Map ─────────────────────────────────────────────
    ax2.imshow(image_rgb, extent=extent, aspect="equal")
    if sign_map is not None:
        vabs = float(np.abs(sign_map).max())
        vabs = vabs if vabs > 1e-10 else 1.0
        # The mask is based on norm_map (not on sign_map==0, which is a legitimate value)
        sign_plot = np.ma.masked_where(no_shap_mask, sign_map)
        im_sign = ax2.imshow(
            sign_plot, extent=extent, aspect="equal",
            cmap="RdBu_r", alpha=alpha_shap, vmin=-vabs, vmax=vabs,
        )
    else:
        map_plot2 = np.ma.masked_where(no_shap_mask, norm_map)
        im_sign = ax2.imshow(
            map_plot2, extent=extent, aspect="equal",
            cmap="RdBu_r", alpha=alpha_shap, vmin=0, vmax=1,
        )
    ax2.set_xlim(0, w); ax2.set_ylim(h, 0)
    ax2.set_title("SHAP Sign Map", fontsize=18, fontweight="bold", pad=8)
    ax2.axis("off")

    # ── Subplot 4: Segmentation ──────────────────────────────────────────────
    ax3.imshow(image, extent=extent, aspect="equal")
    ax3.set_xlim(0, w); ax3.set_ylim(h, 0)

    region_colors = {
        "background_mask": ("black",      0.75),
        "parenchyma_mask": ("darkorange", 0.40),
        "vessels":         ("purple",     0.75),
        "optic_disc_mask": ("red",        0.55),
        "macula_mask":     ("cyan",       0.40),
        "faz_mask":        ("lime",       0.65),
    }
    for region_key, (color, alpha) in region_colors.items():
        mask   = seg[region_key]
        layer  = np.zeros((h, w, 4))
        layer[mask] = to_rgba(color, alpha=alpha)
        ax3.imshow(layer, extent=extent, aspect="equal")

    legend_elements = [
        Patch(facecolor="black",      label="Outside retinal field", edgecolor="gray"),
        Patch(facecolor="darkorange", label="Retinal parenchyma"),
        Patch(facecolor="purple",     label="Blood vessels"),
        Patch(facecolor="red",        label="Optic disc"),
        Patch(facecolor="cyan",       label="Macular-centered ROI"),
        Patch(facecolor="lime",       label="Foveal avascular zone (FAZ)"),
    ]
    ax3.legend(handles=legend_elements, loc="lower right",
               fontsize=11, framealpha=0.88)
    ax3.set_title("Retinal Segmentation", fontsize=18, fontweight="bold", pad=8)
    ax3.axis("off")

    # ── Geometric contours ───────────────────────────────────────────────────
    cx       = seg["cx"];       cy       = seg["cy"]
    macula_x = seg["macula_x"]; macula_y = seg["macula_y"]
    disc_radius   = seg["disc_radius"]
    macula_radius = seg["macula_radius"]
    faz_radius    = seg["faz_radius"]

    circle_styles = {
        "Optic disc":           dict(linestyle="-",  linewidth=1.0),
        "Macular-centered ROI": dict(linestyle="--", linewidth=0.7),
        "FAZ":                  dict(linestyle=":",  linewidth=1.5),
    }

    def add_circle_contour(ax, cx, cy, radius, style, label_str):
        ax.add_patch(plt.Circle(
            (cx, cy), radius, fill=False, edgecolor="black",
            linewidth=style["linewidth"] + 1.2,
            linestyle=style["linestyle"], alpha=0.55,
        ))
        ax.add_patch(plt.Circle(
            (cx, cy), radius, fill=False, edgecolor="white",
            linewidth=style["linewidth"],
            linestyle=style["linestyle"], label=label_str,
        ))

    for ax_i in (ax1, ax2):
        add_circle_contour(ax_i, cx,       cy,       disc_radius,   circle_styles["Optic disc"],           "Optic disc")
        add_circle_contour(ax_i, macula_x, macula_y, macula_radius, circle_styles["Macular-centered ROI"], "Macular-centered ROI")
        add_circle_contour(ax_i, macula_x, macula_y, faz_radius,    circle_styles["FAZ"],                  "FAZ")

    legend_handles_shap = [
        Line2D([0], [0], color="white",
               linewidth=circle_styles[lbl]["linewidth"],
               linestyle=circle_styles[lbl]["linestyle"],
               label=lbl)
        for lbl in ["Optic disc", "Macular-centered ROI", "FAZ"]
    ]
    for ax_i in (ax1, ax2):
        ax_i.legend(
            handles=legend_handles_shap, loc="lower left",
            fontsize=10, framealpha=0.75,
            facecolor="#1a1a1a", edgecolor="white", labelcolor="white",
        )

    # ── Colorbars with fig.add_axes() using real positions ───────────────────
    fig.canvas.draw()

    cbar_w   = 0.012
    cbar_gap = 0.004

    def make_cbar_axes(ax):
        pos = ax.get_position()
        return fig.add_axes([pos.x1 + cbar_gap, pos.y0, cbar_w, pos.height])

    cax1 = make_cbar_axes(ax1)
    cbar1 = fig.colorbar(im_shap, cax=cax1)
    cbar1.set_label("SHAP Importance (norm.)", fontsize=12, fontweight="bold")
    cbar1.ax.tick_params(labelsize=11)
    cbar1.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    cax2 = make_cbar_axes(ax2)
    cbar2 = fig.colorbar(im_sign, cax=cax2)
    cbar2.set_label("SHAP Signed Value", fontsize=12, fontweight="bold")
    cbar2.ax.tick_params(labelsize=11)
    cbar2.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    # ── Suptitle ────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        f"Model: {model_name}  |  Image: {img_name}  |  Prediction: {prediction:.4f}",
        fontsize=11, ha="center", va="bottom",
    )

    figure_path = os.path.join(save_dir, f"{img_name}_{model_name}_shap_segmentation.png")
    plt.savefig(figure_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Figure saved: {figure_path}")
    return figure_path

def analyze_retinal_image(
    img_path: str,
    seg_path: str,
    model_path: str,
    save_dir: str,
    backbone_name: str,
    target_size: tuple = (224, 224),
    alpha_shap: float  = 0.50,
    pred_index: int    = 0,
    background_dir: str | None       = None,
    background_images: np.ndarray | None = None,
    n_background: int  = 20,
    cmap_shap: str     = "jet",
    # ── Batch optimization parameters ───────────────────────────────────────
    preloaded_model=None,          # already-loaded tf.keras.Model (avoids reload)
    preloaded_background: np.ndarray | None = None,  # already-preprocessed background
    save_maps_csv: bool = True,  # False in batch to save disk space
) -> dict:
    """
    Full pipeline: SHAP + segmentation + figure + metrics.

    Parameters
    ----------
    img_path              : Path to the fundus image (with or without extension).
    seg_path               : Path to the vessel segmentation (.png, binary).
    model_path             : Path to the Keras model (.hdf5 / .keras).
    save_dir               : Output directory.
    backbone_name           : Name of the base architecture (see list in the code).
    target_size             : Model input size (width, height).
    alpha_shap              : Transparency of the SHAP overlay.
    pred_index              : Output neuron to explain.
    background_dir          : Folder with background images for SHAP.
    background_images       : np.ndarray (N, H, W, 3) of already-loaded background.
    n_background             : Maximum number of background images.
    cmap_shap                : Colormap of the SHAP map (default 'jet').
    preloaded_model           : Already-loaded model (internal batch use to avoid reload).
    preloaded_background      : Already-preprocessed background (internal batch use).

    Returns
    -------
    dict with keys:
        raw_map, norm_map, metrics, figure_path,
        raw_csv_path, norm_csv_path, metrics_csv_path, prediction
    """
    os.makedirs(save_dir, exist_ok=True)
    model_name = os.path.basename(model_path).split(".")[0]

    # ── Resolve image extension ──
    img_path = resolve_image_path(img_path)
    img_name = os.path.basename(img_path).split(".")[0]

    # ── 1. Model ──
    if preloaded_model is not None:
        model = preloaded_model
    else:
        print(f"\n[1/5] Loading model : {model_name}  ({backbone_name})")
        model = tf.keras.models.load_model(model_path)

    # ── 2. Image ──
    print(f"[2/5] Loading image : {img_name}")
    image_rgb     = load_original_image(img_path)
    img_tensor_np = load_and_preprocess_image(image_rgb, target_size, backbone_name)

    prediction = float(model.predict(img_tensor_np, verbose=0)[0][pred_index])
    prediction = np.abs(1-prediction)
    print(f"      Prediction [{pred_index}]: {prediction:.4f}")

    # ── 3. SHAP ──
    print(f"[3/5] Computing SHAP map...")

    if preloaded_background is not None:
        bg = preloaded_background
        print(f"  Using preloaded background ({len(bg)} imgs).")
    elif background_dir is not None:
        print(f"  Loading background from: {background_dir}")
        bg = load_background_from_folder(
            background_dir, target_size, backbone_name, n_background
        )
    elif background_images is not None:
        print(f"  Using provided background_images ({len(background_images)} imgs).")
        bg = background_images.astype(np.float32)
        if bg.shape[1:3] != target_size:
            imgs_rgb = [cv2.resize(b.astype(np.uint8), target_size) for b in bg]
            bg = np.array([
                load_and_preprocess_image(i, target_size, backbone_name)[0]
                for i in imgs_rgb
            ], dtype=np.float32)
    else:
        print("  No background -> using black images (fallback).")
        bg = np.zeros((n_background, *target_size, 3), dtype=np.float32)

    raw_map, raw_sign_map = deep_shap(model, img_tensor_np, bg, pred_index)
    print(f"  SHAP map shape (model space): {raw_map.shape}")

    # ── Resize to the original size ──
    h_orig, w_orig = image_rgb.shape[:2]
    raw_map_orig = cv2.resize(
        raw_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    sign_map_orig = cv2.resize(
        raw_sign_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    norm_map = normalize_map(raw_map_orig)

    # ── Save maps (optional, skip in batch to save disk space) ──
    raw_csv_path  = None
    norm_csv_path = None
    sign_csv_path = None
    if save_maps_csv:
        raw_csv_path  = os.path.join(save_dir, f"{img_name}_{model_name}_shap_raw.csv")
        norm_csv_path = os.path.join(save_dir, f"{img_name}_{model_name}_shap_norm.csv")
        sign_csv_path = os.path.join(save_dir, f"{img_name}_{model_name}_shap_sign.csv")
        pd.DataFrame(raw_map_orig).to_csv(raw_csv_path,  index=False, header=False)
        pd.DataFrame(norm_map).to_csv(norm_csv_path,      index=False, header=False)
        pd.DataFrame(sign_map_orig).to_csv(sign_csv_path, index=False, header=False)
        print(f"  Raw CSV    : {raw_csv_path}")
        print(f"  Norm CSV   : {norm_csv_path}")
        print(f"  Sign CSV   : {sign_csv_path}")

    # ── 4. Segmentation ──
    print(f"[4/5] Segmenting retina...")
    seg = segment_retina(img_path, seg_path)

    print("  Pixels per region:")
    for key in ("vessels", "optic_disc_mask", "faz_mask", "macula_mask", "parenchyma_mask", "background_mask"):
        print(f"    {key:<15}: {seg[key].sum():>10,}")

    # ── 5. Metrics ──
    print(f"[5/5] Computing SHAP metrics...")
    metrics = calculate_shap_metrics(norm_map, seg, sign_map=sign_map_orig)

    print(f"\n  Center of mass   : ({metrics['center_of_mass_x']}, {metrics['center_of_mass_y']})")
    print(f"  Entropy          : {metrics['entropy']:.4f}")
    print(f"  Mean importance  : {metrics['mean_importance']:.4f}")
    print(f"  Std importance   : {metrics['std_importance']:.4f}")
    print(f"  Gini coefficient : {metrics['gini_coefficient']:.4f}")
    print(f"  Top-5%% area      : {metrics['top5pct_area']:.2f}%")
    if "signed_mean" in metrics:
        print(f"\n  --- Signed SHAP (global) ---")
        print(f"  Signed mean      : {metrics['signed_mean']:.6f}  (>0 → pushes towards positive class)")
        print(f"  Pct positive px  : {metrics['pct_positive_pixels']:.2f}%")
        print(f"  Pct negative px  : {metrics['pct_negative_pixels']:.2f}%")
        print(f"  Net signed sum   : {metrics['net_signed_sum']:.4f}")
    print(f"\n  SHAP by region:")
    for reg, vals in metrics["shap_by_region"].items():
        print(f"    {reg:<12}: {vals['pct_of_total_shap']:>6.2f}% SHAP  |  "
              f"{vals['pct_of_image_area']:>6.2f}% area  |  "
              f"ratio={vals['shap_area_ratio']:.3f}  |  "
              f"mean={vals['shap_mean']:.5f}", end="")
        if "signed_mean" in vals:
            print(f"  |  signed_mean={vals['signed_mean']:+.5f}  dir={vals['net_direction']}", end="")
        print()

    # ── Metrics CSV per image (skip in batch; the batch has its own cumulative CSV) ──
    metrics_csv_path = None
    if save_maps_csv:
        metrics_csv_path = os.path.join(save_dir, f"{img_name}_{model_name}_metrics.csv")
        rows = []
        global_row = {
            "image":                  img_name,
            "model":                  model_name,
            "prediction":             round(prediction, 6),
            "center_of_mass_x":       metrics["center_of_mass_x"],
            "center_of_mass_y":       metrics["center_of_mass_y"],
            # ── Center-of-mass location ──
            "com_in_faz":             metrics["com_in_faz"],
            "com_in_macula":          metrics["com_in_macula"],
            "entropy":                metrics["entropy"],
            "mean_importance":        metrics["mean_importance"],
            "max_importance":         metrics["max_importance"],
            "std_importance":         metrics["std_importance"],
            "gini_coefficient":       metrics["gini_coefficient"],
            "top5pct_area":           metrics["top5pct_area"],
            "top5pct_mean_importance":metrics["top5pct_mean_importance"],
            # ── Global sign metrics ──
            "signed_mean":            metrics.get("signed_mean", None),
            "signed_std":             metrics.get("signed_std", None),
            "pct_positive_pixels":    metrics.get("pct_positive_pixels", None),
            "pct_negative_pixels":    metrics.get("pct_negative_pixels", None),
            "net_signed_sum":         metrics.get("net_signed_sum", None),
        }
        for region, vals in metrics["shap_by_region"].items():
            global_row[f"{region}_shap_sum"]          = vals["shap_sum"]
            global_row[f"{region}_shap_mean"]         = vals["shap_mean"]
            global_row[f"{region}_shap_std"]          = vals["shap_std"]
            global_row[f"{region}_pct_of_total_shap"] = vals["pct_of_total_shap"]
            global_row[f"{region}_pct_of_image_area"] = vals["pct_of_image_area"]
            global_row[f"{region}_shap_area_ratio"]   = vals["shap_area_ratio"]
            global_row[f"{region}_n_pixels"]          = vals["n_pixels"]
            # per-region sign columns
            global_row[f"{region}_signed_mean"]         = vals.get("signed_mean", None)
            global_row[f"{region}_signed_sum"]          = vals.get("signed_sum", None)
            global_row[f"{region}_pct_positive_pixels"] = vals.get("pct_positive_pixels", None)
            global_row[f"{region}_pct_negative_pixels"] = vals.get("pct_negative_pixels", None)
            global_row[f"{region}_net_direction"]       = vals.get("net_direction", None)
        rows.append(global_row)
        pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)
        print(f"\n  Metrics CSV: {metrics_csv_path}")

    # ── Figure ──
    figure_path = save_combined_figure(
        image_rgb    = image_rgb,
        norm_map     = norm_map,
        sign_map     = sign_map_orig,
        seg          = seg,
        metrics      = metrics,
        img_name     = img_name,
        model_name   = model_name,
        prediction   = prediction,
        save_dir     = save_dir,
        alpha_shap   = alpha_shap,
        cmap_shap    = cmap_shap,
    )

    return {
        "raw_map":             raw_map_orig,
        "norm_map":            norm_map,
        "sign_map":            sign_map_orig,
        "metrics":             metrics,
        "figure_path":         figure_path,
        "raw_csv_path":        raw_csv_path,
        "norm_csv_path":       norm_csv_path,
        "sign_csv_path":       sign_csv_path,
        "metrics_csv_path":    metrics_csv_path,
        "prediction":          prediction,
    }

def analyze_batch(
    img_dir:        str,
    seg_dir:        str,
    model_path:     str,
    save_dir:       str,
    backbone_name:  str,
    background_dir: str,
    n_background:   int   = 500,
    target_size:    tuple = (224, 224),
    pred_index:     int   = 0,
    alpha_shap:     float = 0.50,
    cmap_shap:      str   = "jet",
) -> None:
    """
    Processes all images in img_dir and saves:
      - One PNG figure per image in  save_dir/figures/
      - A cumulative CSV in          save_dir/batch_results.csv
      - An error log in              save_dir/errors.log

    Automatic resume: if the CSV already exists and contains a row with the
    same image name + model, that image is skipped without reprocessing.

    Parameters
    ----------
    img_dir        : Folder with the images to analyze (.jpg/.jpeg/.png/...).
    seg_dir        : Folder with the vessel segmentations (.png, same name).
    model_path     : Path to the Keras model (.hdf5 / .keras).
    save_dir       : Root output directory.
    backbone_name  : Name of the architecture (see list in analyze_retinal_image).
    background_dir : Folder with background images for SHAP.
    n_background   : Number of background images (default 500).
    target_size    : Model input size.
    pred_index     : Output neuron to explain.
    alpha_shap     : Transparency of the SHAP overlay.
    cmap_shap      : Colormap of the SHAP map.
    """
    import traceback
    import datetime

    # ── Output directories ──────────────────────────────────────────────────
    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "batch_results.csv")
    log_path = os.path.join(save_dir, "errors.log")
    model_name = os.path.basename(model_path).split(".")[0]

    # ── Already-processed images (to resume) ────────────────────────────────
    processed: set[str] = set()
    if os.path.isfile(csv_path):
        df_prev = pd.read_csv(csv_path, usecols=["image", "model"])
        processed = set(
            df_prev[df_prev["model"] == model_name]["image"].astype(str)
        )
        print(f"[BATCH] Existing CSV: {len(processed)} image(s) already processed, will be skipped.")

    # ── List images to process ──────────────────────────────────────────────
    all_paths = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in img_extensions
    ])

    pending = [
        r for r in all_paths
        if os.path.splitext(os.path.basename(r))[0] not in processed
    ]

    total     = len(all_paths)
    n_skip    = total - len(pending)
    n_pending = len(pending)

    print(f"[BATCH] Images found      : {total}")
    print(f"[BATCH] Already processed : {n_skip}")
    print(f"[BATCH] To process        : {n_pending}")

    if n_pending == 0:
        print("[BATCH] Nothing to process. Done.")
        return

    # ── Single load of model and background ─────────────────────────────────
    print(f"\n[BATCH] Loading model: {model_name} ({backbone_name})")
    model = tf.keras.models.load_model(model_path)

    print(f"[BATCH] Loading {n_background} background images from: {background_dir}")
    bg = load_background_from_folder(
        background_dir, target_size, backbone_name, n_background
    )
    print(f"[BATCH] Background ready: {bg.shape}")

    # ── Main loop ────────────────────────────────────────────────────────────
    errors = []
    for idx, img_path_item in enumerate(pending, start=1):
        img_name = os.path.splitext(os.path.basename(img_path_item))[0]
        seg_path = os.path.join(seg_dir, img_name + ".png")
        start_ts = datetime.datetime.now()

        print(f"\n{'─'*70}")
        print(f"[{idx}/{n_pending}]  {img_name}  ({start_ts.strftime('%H:%M:%S')})")

        # ── Check that the segmentation exists ──
        if not os.path.isfile(seg_path):
            msg = f"Segmentation not found: {seg_path}"
            print(f"  [SKIP] {msg}")
            errors.append({"image": img_name, "error": msg})
            log_error(log_path, img_name, msg)
            continue

        try:
            result = analyze_retinal_image(
                img_path         = img_path_item,
                seg_path         = seg_path,
                model_path       = model_path,
                save_dir         = figures_dir,
                backbone_name    = backbone_name,
                target_size      = target_size,
                alpha_shap       = alpha_shap,
                pred_index       = pred_index,
                cmap_shap        = cmap_shap,
                preloaded_model     = model,
                preloaded_background = bg,
                save_maps_csv    = False,   # the cumulative CSV is handled by the batch
            )

            # ── Append to the cumulative CSV ──────────────────────────────
            append_csv_row(csv_path, result["metrics"], img_name,
                             model_name, result["prediction"])

            elapsed = (datetime.datetime.now() - start_ts).total_seconds()
            print(f"  [OK]  {img_name}  —  {elapsed:.1f}s")

        except Exception as exc:
            tb = traceback.format_exc()
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  [ERROR] {msg}")
            errors.append({"image": img_name, "error": msg})
            log_error(log_path, img_name, tb)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"[BATCH] Completed. Processed: {n_pending - len(errors)}/{n_pending}")
    if errors:
        print(f"[BATCH] Errors ({len(errors)}): see {log_path}")
        for e in errors:
            print(f"    ✗  {e['image']}: {e['error']}")
    print(f"[BATCH] Cumulative CSV : {csv_path}")
    print(f"[BATCH] Figures        : {figures_dir}")

def append_csv_row(csv_path: str, metrics: dict, img_name: str,
                     model_name: str, prediction: float) -> None:
    """Appends a row to the cumulative CSV (creates the file if it doesn't exist)."""
    row = {
        "image":                   img_name,
        "model":                   model_name,
        "prediction":              round(prediction, 6),
        "center_of_mass_x":        metrics["center_of_mass_x"],
        "center_of_mass_y":        metrics["center_of_mass_y"],
        "com_in_faz":              metrics["com_in_faz"],
        "com_in_macula":           metrics["com_in_macula"],
        "entropy":                 metrics["entropy"],
        "mean_importance":         metrics["mean_importance"],
        "max_importance":          metrics["max_importance"],
        "std_importance":          metrics["std_importance"],
        "gini_coefficient":        metrics["gini_coefficient"],
        "top5pct_area":            metrics["top5pct_area"],
        "top5pct_mean_importance": metrics["top5pct_mean_importance"],
        # ── Global sign metrics ──
        "signed_mean":             metrics.get("signed_mean", None),
        "signed_std":              metrics.get("signed_std", None),
        "pct_positive_pixels":     metrics.get("pct_positive_pixels", None),
        "pct_negative_pixels":     metrics.get("pct_negative_pixels", None),
        "net_signed_sum":          metrics.get("net_signed_sum", None),
    }
    for region, vals in metrics["shap_by_region"].items():
        row[f"{region}_shap_sum"]          = vals["shap_sum"]
        row[f"{region}_shap_mean"]         = vals["shap_mean"]
        row[f"{region}_shap_std"]          = vals["shap_std"]
        row[f"{region}_pct_of_total_shap"] = vals["pct_of_total_shap"]
        row[f"{region}_pct_of_image_area"] = vals["pct_of_image_area"]
        row[f"{region}_shap_area_ratio"]   = vals["shap_area_ratio"]
        row[f"{region}_n_pixels"]          = vals["n_pixels"]
        # per-region sign columns
        row[f"{region}_signed_mean"]         = vals.get("signed_mean", None)
        row[f"{region}_signed_sum"]          = vals.get("signed_sum", None)
        row[f"{region}_pct_positive_pixels"] = vals.get("pct_positive_pixels", None)
        row[f"{region}_pct_negative_pixels"] = vals.get("pct_negative_pixels", None)
        row[f"{region}_net_direction"]       = vals.get("net_direction", None)

    new_df = pd.DataFrame([row])
    write_header = not os.path.isfile(csv_path)
    new_df.to_csv(csv_path, mode="a", header=write_header, index=False)

def log_error(log_path: str, img_name: str, details: str) -> None:
    """Appends an entry to the error log."""
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{ts}]  {img_name}\n")
        f.write(details + "\n")


if __name__ == "__main__":

    analyze_batch(
        img_dir        = r"",
        seg_dir        = r"",
        model_path     = r"",
        save_dir       = r"",
        backbone_name  = "InceptionResNetV21",
        background_dir = r"",
        pred_index     = 0,
        alpha_shap     = 0.50,
        cmap_shap      = "jet",
    )
