"""
Crokinole Score Detection System
Core functions for board detection, disc detection, and scoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color, feature, transform, draw, filters, measure, morphology, exposure
from matplotlib import patches
from dataclasses import dataclass

def _ring_radius_with_fallback(detected_rings, name, outer_r, ring_ratios):
    """
    Return radius for ring `name`.

    If that ring wasn't detected, approximate it as:
        outer_r * ring_ratios[name]
    where ring_ratios comes from CONFIG['ring_ratios'].
    """
    r = detected_rings.get(name, None)
    if r is None and outer_r is not None:
        ratio = ring_ratios.get(name, None)
        if ratio is not None:
            r = int(outer_r * ratio)
    return r

def display_image(img, title="Image", figsize=(8, 8)):
    """Display a single image with optional title."""
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def display_images(images, titles=None, figsize=(14, 7)):
    """Display multiple images side by side."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def display_transform_info(transform_info, n_rings):
    """Display information about perspective transformation."""
    print(f"Perspective correction computed from {n_rings * 50} point correspondences")
    
    if transform_info['applied']:
        print(f"\nTransformation applied:")
        print(f"  Overall magnitude: {transform_info['magnitude']:.4f}")
        print(f"  Perspective distortion: {transform_info['perspective_strength']:.4f}")
        if transform_info['perspective_strength'] > 0.001:
            print(f"  → Significant perspective correction applied")
        else:
            print(f"  → Minor adjustments only (near-orthogonal view)")
    else:
        print(f"\nNo significant transformation needed (image already orthogonal)")


def detect_edges(img, config):
    """Detect edges using Canny edge detection."""
    board_cfg = config['board_detection']

    # Handle grayscale, RGB, and RGBA safely
    if img.ndim == 2:
        # Already grayscale
        gray_image = img
    else:
        # If image has an alpha channel, drop/convert it
        if img.shape[-1] == 4:
            # Option A: use rgba2rgb to blend with white background
            img_rgb = color.rgba2rgb(img)
            # Option B (simpler): img_rgb = img[..., :3]
        else:
            img_rgb = img

        gray_image = color.rgb2gray(img_rgb)

    edges = feature.canny(
        gray_image,
        low_threshold=board_cfg['canny_low_threshold'],
        high_threshold=board_cfg['canny_high_threshold'],
        sigma=board_cfg['edge_sigma']
    )
    return edges



def detect_board_and_rings(edges, config):
    """Detect outer board circle and inner rings. Returns None if not a valid board."""
    h, w = edges.shape[:2]
    
    board_cfg = config['board_detection']
    ring_ratios = config['ring_ratios']
    ring_search_cfg = config['ring_search']
    
    # Detect outer board circle
    min_radius = int(min(h, w) * board_cfg['min_circle_ratio'])
    max_radius = int(min(h, w) * board_cfg['max_circle_ratio'])
    radii = np.arange(min_radius, max_radius, board_cfg['radius_step'])
    
    hough_res = transform.hough_circle(edges, radii)
    accums, cx, cy, radii_detected = transform.hough_circle_peaks(hough_res, radii, total_num_peaks=1)
    
    if len(cx) == 0:
        return None
    
    board_center = (int(cx[0]), int(cy[0]))
    board_radius = int(radii_detected[0])
    
    # Detect inner rings
    min_search_r = int(board_radius * 0.05)
    max_search_r = int(board_radius * 0.98)
    search_radii = np.arange(min_search_r, max_search_r, ring_search_cfg['step_size'])
    
    hough_res = transform.hough_circle(edges, search_radii)
    accums, cx, cy, radii_det = transform.hough_circle_peaks(hough_res, search_radii, total_num_peaks=30)
    
    # Filter candidates near board center
    max_offset = board_radius * ring_search_cfg['max_center_offset']
    candidates = []
    for x, y, r, acc in zip(cx, cy, radii_det, accums):
        dist = np.hypot(x - board_center[0], y - board_center[1])
        if dist < max_offset:
            candidates.append({
                'radius': r,
                'center': (x, y),
                'accumulator': acc
            })
    
    candidates.sort(key=lambda x: x['radius'])
    
    detected_rings = {'outer': board_radius}
    used_candidates = set()
    
    # Match candidates to expected ring ratios
    inner_ratios = {k: v for k, v in ring_ratios.items() if k != 'outer'}
    sorted_rings = sorted(inner_ratios.items(), key=lambda x: x[1], reverse=True)
    
    for ring_name, ratio in sorted_rings:
        expected_r = int(board_radius * ratio)
        tol = expected_r * ring_search_cfg['tolerance']
        
        best_match = None
        best_diff = float('inf')
        best_idx = None
        for idx, c in enumerate(candidates):
            if idx in used_candidates:
                continue
            diff = abs(c['radius'] - expected_r)
            if diff < tol and diff < best_diff:
                best_diff = diff
                best_match = c
                best_idx = idx
        
        if best_match is not None:
            detected_rings[ring_name] = best_match['radius']
            used_candidates.add(best_idx)
    
    # Always require at least an outer circle + *one* inner ring
    if len(detected_rings) <= 1:
        # Only outer ring found → probably not a board
        return None

    # Otherwise, return whatever rings we managed to detect.
    # The caller (validation step) will decide if it's "good enough".
    return {
        'center': board_center,
        'radius': board_radius,
        'rings': detected_rings
    }


def visualize_board_detection(img, board_result):
    """Visualize detected board and rings with color-coded circles."""
    if board_result is None:
        return
    
    board_center = board_result['center']
    detected_rings = board_result['rings']
    
    colors = {
        'outer':  [0, 255, 0],
        'ring_15':[255, 255, 0],
        'ring_10':[255, 165, 0],
        'ring_5': [255, 0, 0],
        'center': [0, 0, 255]
    }
    
    # --- NEW: ensure overlay is 3-channel RGB ---
    if img.ndim == 2:
        # grayscale → stack to 3 channels
        base = np.stack([img, img, img], axis=-1)
    else:
        if img.shape[-1] == 4:
            # RGBA → drop alpha
            base = img[..., :3]
        else:
            base = img
    overlay = base.copy()
    # -------------------------------------------
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    
    for ring_name, ring_radius in detected_rings.items():
        if ring_name in colors:
            circy, circx = draw.circle_perimeter(
                board_center[1], board_center[0], ring_radius,
                shape=overlay.shape[:2]
            )
            overlay[circy, circx] = colors[ring_name]
    
    ax.imshow(overlay)
    ax.plot(board_center[0], board_center[1], 'ro', markersize=8)
    ax.set_title("Detected Board and Rings")
    ax.axis('off')
    plt.show()



def correct_perspective(img, board_result):
    """
    Correct perspective distortion by registering detected rings to perfect circles.
    
    The detected rings are actually ellipses due to perspective, but Hough circles
    fit circles to them. We sample points from these detected "circles" (which follow
    the elliptical edges) and map them to perfect circles at the same radii.
    """
    if board_result is None:
        return None, None
    
    detected_center = board_result['center']
    detected_rings = board_result['rings']
    
    # Sample points around each detected ring
    def sample_ring_points(center, radius, n_points=50):
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        points = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    # Collect point correspondences from all rings
    src_points = []  # Detected positions (elliptical in reality)
    dst_points = []  # Target positions (perfect circles)
    
    for ring_name, radius in detected_rings.items():
        # Source: sample points from detected ring (follows ellipse due to perspective)
        ring_src = sample_ring_points(detected_center, radius, n_points=50)
        src_points.append(ring_src)
        
        # Destination: same points but on perfect circle (orthogonal view)
        ring_dst = sample_ring_points(detected_center, radius, n_points=50)
        dst_points.append(ring_dst)
    
    # Combine all points from all rings
    src_points = np.vstack(src_points)
    dst_points = np.vstack(dst_points)
    
    # Compute projective transformation (homography)
    # This maps the elliptical rings → circular rings
    tform = transform.ProjectiveTransform()
    tform.estimate(src_points, dst_points)
    
    # Analyze transformation magnitude
    # Compare identity matrix to actual transformation
    identity = np.eye(3)
    diff = np.linalg.norm(tform.params - identity)
    
    # Check for perspective distortion (non-zero off-diagonal elements in last row)
    perspective_strength = np.linalg.norm(tform.params[2, :2])
    
    transform_info = {
        'applied': diff > 0.01,  # Significant transformation applied
        'magnitude': diff,
        'perspective_strength': perspective_strength,
        'matrix': tform.params
    }
    
    # Warp the image to correct perspective
    straightened = transform.warp(img, tform.inverse, output_shape=img.shape)
    
    # Convert back to uint8 for display
    straightened = (straightened * 255).astype(np.uint8)
    
    return straightened, transform_info


def create_scoring_regions(img_shape, board_result, config):
    """
    Create scoring mask (0,5,10,15,20) using detected rings
    plus ratio-based fallbacks from `config`.
    Radial order (outer → center): outer > ring_5 > ring_10 > ring_15 > center.
    """
    if board_result is None:
        return None

    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    board_center   = board_result['center']
    detected_rings = board_result['rings']
    ring_ratios    = config.get('ring_ratios', {})

    # --- robust radii with fallback to CONFIG ratios ------------------------
    outer_r = detected_rings.get('outer', None)
    if outer_r is None:
        return None

    ring_5_r  = _ring_radius_with_fallback(detected_rings, 'ring_5',  outer_r, ring_ratios)
    ring_10_r = _ring_radius_with_fallback(detected_rings, 'ring_10', outer_r, ring_ratios)
    ring_15_r = _ring_radius_with_fallback(detected_rings, 'ring_15', outer_r, ring_ratios)
    center_r  = _ring_radius_with_fallback(detected_rings, 'center',  outer_r, ring_ratios)

    # enforce monotone ordering: ring_5 > ring_10 > ring_15 > center
    radii = [r for r in [ring_5_r, ring_10_r, ring_15_r, center_r] if r is not None]
    if len(radii) >= 2:
        radii_sorted = sorted(radii, reverse=True)  # biggest → smallest
        i = 0
        if ring_5_r  is not None: ring_5_r  = radii_sorted[i]; i += 1
        if ring_10_r is not None: ring_10_r = radii_sorted[i]; i += 1
        if ring_15_r is not None: ring_15_r = radii_sorted[i]; i += 1
        if center_r  is not None and i < len(radii_sorted):
            center_r = radii_sorted[i]

    # -----------------------------------------------------------------------
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - board_center[0])**2 + (y - board_center[1])**2)

    # 5 pt: between ring_5 and ring_10
    if ring_5_r is not None and ring_10_r is not None:
        mask[(distances <= ring_5_r) & (distances > ring_10_r)] = 5

    # 10 pt: between ring_10 and ring_15
    if ring_10_r is not None and ring_15_r is not None:
        mask[(distances <= ring_10_r) & (distances > ring_15_r)] = 10

    # 15 pt: between ring_15 and center
    if ring_15_r is not None and center_r is not None:
        mask[(distances <= ring_15_r) & (distances > center_r)] = 15

    # 20 pt: inside center hole
    if center_r is not None:
        mask[distances <= center_r] = 20

    return mask

def _angular_uniformity(lum, x, y, r, n=36, frac=0.5):
    """
    Sample grayscale values on a mid-radius circle (frac * r) around (x,y).
    Returns (std, mean). Real discs should have low std (nearly uniform colour).
    """
    h, w = lum.shape
    if r <= 2:
        return 0.0, 0.0
    radius = max(2.0, frac * float(r))
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.clip((x + radius * np.cos(angles)).round().astype(int), 0, w-1)
    ys = np.clip((y + radius * np.sin(angles)).round().astype(int), 0, h-1)
    vals = lum[ys, xs]
    return float(vals.std()), float(vals.mean())



def visualize_scoring_regions(img, scoring_mask):
    """
    Visualize the scoring regions with colored overlay.
    Each scoring region gets a distinct color.
    """
    if scoring_mask is None:
        print("No scoring mask to visualize")
        return

    # ---------- NEW: normalize img to RGB (H,W,3) ----------
    if img.ndim == 2:
        # grayscale -> stack
        base = np.stack([img, img, img], axis=-1)
    else:
        if img.shape[-1] == 4:
            # RGBA -> drop alpha
            base = img[..., :3]
        else:
            base = img
    # make sure it's uint8 for display
    if base.dtype != np.uint8:
        base = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    # -------------------------------------------------------

    # Define colors for each scoring region (RGB)
    region_colors = {
        0: [50, 50, 50],      # Outside - dark gray
        5: [100, 200, 100],   # Outer ring (5pt) - light green
        10: [255, 200, 100],  # Middle ring (10pt) - orange
        15: [255, 255, 100],  # Inner ring (15pt) - yellow
        20: [255, 100, 100]   # Center (20pt) - red
    }

    # Create colored overlay
    overlay = np.zeros((*scoring_mask.shape, 3), dtype=np.uint8)
    for score_value, rgb_color in region_colors.items():
        mask_region = scoring_mask == score_value
        overlay[mask_region] = rgb_color

    # Blend with original image (now base is H×W×3)
    alpha = 0.5
    blended = (alpha * base + (1 - alpha) * overlay).astype(np.uint8)

    # Display
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(base)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title("Scoring Regions Mask")
    axes[1].axis('off')

    axes[2].imshow(blended)
    axes[2].set_title("Blended Overlay")
    axes[2].axis('off')

    # Legend + stats (unchanged)
    unique_scores = np.unique(scoring_mask)
    legend_text = "Scoring Regions:\n"
    for score in sorted(unique_scores):
        if score == 0:
            legend_text += f"  {score}pt: Outside board\n"
        elif score == 20:
            legend_text += f"  {score}pt: Center hole\n"
        else:
            legend_text += f"  {score}pt: Ring\n"

    plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    print(f"Scoring regions: {sorted(unique_scores)} points")
    for score in sorted(unique_scores):
        count = np.sum(scoring_mask == score)
        percentage = (count / scoring_mask.size) * 100
        print(f"  {score}pt region: {count} pixels ({percentage:.1f}%)")

# -----------------------------
# STEP 5: Disc detection + colour grouping
# -----------------------------

def _expected_disc_radius(board_result, config, pad_frac=0.10, min_px=4):
    """
    Use the 'center' ring if present; otherwise approximate it from outer radius
    and CONFIG['ring_ratios']['center'].
    """
    rings = board_result['rings']
    r0 = float(rings.get('center', 0.0))

    if r0 <= 0.0:
        outer_r = float(rings.get('outer', 0.0))
        if outer_r > 0:
            center_ratio = config.get('ring_ratios', {}).get('center', 0.05)
            r0 = outer_r * center_ratio

    if r0 <= 0.0:
        return None, None, None

    band = max(min_px, r0 * pad_frac)
    return r0, max(2, int(r0 - band)), int(r0 + band)



def _play_area_mask(shape, board_result, inset_px=4):
    """
    Inside the playable wood (<= ring_5). If ring_5 is missing, use 0.95*outer.
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = board_result['center']
    rings = board_result['rings']
    rout = float(rings.get('outer', 0.0))
    r5   = float(rings.get('ring_5', 0.95 * rout if rout > 0 else 0.0))

    r_play_outer = r5 if r5 > 0 else rout
    if rout > 0 and r5 > 0:
        r_play_outer = min(r5, rout)

    r_play_outer = max(0.0, r_play_outer - inset_px)
    dist = np.hypot(xx - cx, yy - cy)
    return dist <= r_play_outer


def _edge_ring(edg, x, y, r, n=48):
    """Mean edge value along a circle."""
    h, w = edg.shape
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.clip((x + r*np.cos(ang)).round().astype(int), 0, w-1)
    ys = np.clip((y + r*np.sin(ang)).round().astype(int), 0, h-1)
    return float(edg[ys, xs].mean())


def _inside_stats(lum, x, y, r):
    """Mean/std inside disc, and mean in thin outer ring."""
    h, w = lum.shape
    yy, xx = np.ogrid[:h, :w]
    d = np.hypot(xx - x, yy - y)
    inside = d <= (r - 1)
    ring   = (d > (r + 1)) & (d <= (r + 4))
    if not inside.any() or not ring.any():
        return 0.0, 1.0, 0.0
    mu_in  = float(lum[inside].mean())
    sd_in  = float(lum[inside].std())
    mu_out = float(lum[ring].mean())
    return mu_in, sd_in, mu_out


def _lab_patch_mean(lab_img, x, y, half=3):
    """Mean Lab colour in a small square patch around (x,y)."""
    h, w = lab_img.shape[:2]
    x0 = max(0, x - half); x1 = min(w, x + half + 1)
    y0 = max(0, y - half); y1 = min(h, y + half + 1)
    patch = lab_img[y0:y1, x0:x1, :]
    return np.mean(patch.reshape(-1, 3), axis=0)


def _kmeans2_lab(feats, seed=0):
    """
    Robust K=2 k-means on Lab features.
    Handles edge cases: 0 discs, 1 disc, or all discs identical color.
    """
    # Case 0: No discs
    if len(feats) == 0:
        return np.array([], dtype=int), np.zeros((2, 3))

    X = np.asarray(feats, float)
    
    # Case 1: Only one disc found
    # We cannot cluster 1 point into 2 groups. Assign it to group 0.
    if len(X) == 1:
        # Return label [0] and duplicate centroids
        return np.array([0], dtype=int), np.stack([X[0], X[0]])

    rng = np.random.default_rng(seed)

    # --- K-Means++ Initialization (Safe Mode) ---
    # 1. Pick first centroid randomly
    idx0 = rng.integers(0, len(X))
    c0 = X[idx0]

    # 2. Pick second centroid based on distance
    d2 = np.sum((X - c0) ** 2, axis=1)
    total_dist = d2.sum()

    # SAFETY CHECK: If all discs are identical color (total_dist is 0),
    # we can't divide by zero. Just pick a random second point.
    if total_dist < 1e-9:
        idx1 = rng.integers(0, len(X))
    else:
        # Normalize probabilities strictly to sum to 1.0
        probs = d2 / total_dist
        probs = probs / probs.sum() # Double normalization fixes floating point drift
        idx1 = rng.choice(len(X), p=probs)

    c1 = X[idx1]
    C = np.stack([c0, c1], axis=0)

    # --- Lloyd's Algorithm (Standard) ---
    for _ in range(20):
        # Assign points to nearest centroid
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        L = np.argmin(d, axis=1)
        
        C_new = []
        # Update centroids
        for k in (0, 1):
            if np.any(L == k):
                C_new.append(X[L == k].mean(axis=0))
            else:
                # If a cluster became empty, keep old centroid
                C_new.append(C[k])
        
        C_new = np.stack(C_new, axis=0)
        if np.allclose(C_new, C, atol=1e-4):
            break
        C = C_new

    return L.astype(int), C

from skimage import measure, morphology

from skimage import morphology, measure, color, exposure, filters, feature
import numpy as np

from skimage import color, exposure, filters, feature, morphology, measure


# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------
def detect_discs(straightened_img, board_result, config, debug_stats=False):
    """
    Best-of-both-worlds disc detection.

    - WOOD boards: segmentation-based:
        * Dark branch: Strong DARK_DELTA + local annulus contrast (working well).
        * Light branch: **REVISED** Lower LIGHT_DELTA + **NEW local annulus contrast**
          (fixes missing light discs and kills light ghosts).
    """
    h, w = straightened_img.shape[:2]

    # ---------- basic geometry ----------
    r0, r_min, r_max = _expected_disc_radius(board_result, config, pad_frac=0.50)
    if r0 is None:
        print("[DD] No center ring radius; cannot infer disc size.")
        return []

    cx_board, cy_board = board_result['center']
    rings = board_result['rings']

    # 20-hole radius (fallback if missing)
    r_center = float(rings.get('center', 0.0))
    if r_center <= 0.0:
        r_outer = float(rings.get('outer', min(h, w) / 2.0))
        r_center = 0.06 * r_outer

    cfg = config.get('disc_detection', {})
    max_discs = cfg.get('max_discs', 28)

    # ---------- board type ----------
    raw_gray = color.rgb2gray(straightened_img.astype(float) / 255.0)
    board_med_raw = float(np.median(raw_gray))
    is_wood = board_med_raw < 0.65
    print(f"[DD] Board median={board_med_raw:.3f} → {'WOOD' if is_wood else 'WHITE'} mode")

    # ---------- playable mask ----------
    play_mask = _play_area_mask(
        straightened_img.shape,
        board_result,
        inset_px=max(3, int(0.003 * min(h, w)))
    )

    # ---------- white-board path ----------
    if not is_wood:
        return _detect_discs_hough_white(
            straightened_img, board_result, config,
            play_mask, r0, r_min, r_max, r_center, max_discs, debug_stats
        )

    # -----------------------------------------------------------------
    # WOOD BOARD PATH  (best of both worlds)
    # -----------------------------------------------------------------
    img_f = straightened_img.astype(float) / 255.0
    sharp = filters.unsharp_mask(img_f, radius=2, amount=1.5)
    lum   = color.rgb2gray(sharp)
    lum   = exposure.equalize_adapthist(lum, clip_limit=0.01)

    board_med = float(np.median(lum[play_mask]))  # use only playable area

    disc_area = np.pi * (r0 ** 2)
    AREA_MIN  = 0.35 * disc_area
    AREA_MAX  = 1.80 * disc_area
    CIRC_MIN  = 0.60

    # thresholds
    DARK_DELTA   = 0.20   # very dark → black discs
    # LIGHT_DELTA adjusted from 0.03 to 0.01 to catch subtle light discs
    LIGHT_DELTA  = 0.005   
    LOCAL_DELTA_D = 0.10  # local contrast for dark discs
    # NEW: local contrast threshold for light discs to kill glare ghosts
    LOCAL_DELTA_L = 0.05 

    # --- raw masks ---
    dark_raw  = (lum < board_med - DARK_DELTA) & play_mask
    light_raw = (lum > board_med + LIGHT_DELTA) & play_mask

    # --- morphology ---
    se = morphology.disk(max(2, int(0.35 * r0)))
    dark_mask  = morphology.binary_closing(
                    morphology.binary_opening(dark_raw,  se), se
                 )
    light_mask = morphology.binary_closing(
                    morphology.binary_opening(light_raw, se), se
                 )

    # we’ll use union for annulus exclusion later
    disc_mask_union = dark_mask | light_mask

    # convenience grid for annulus computation
    yy, xx = np.mgrid[0:h, 0:w]

    # -------------------------------------------------------------
    # helper: extract dark discs → uses LOCAL annulus contrast
    # -------------------------------------------------------------
    def extract_dark_regions():
        labels = measure.label(dark_mask)
        regions = measure.regionprops(labels, intensity_image=lum)
        out = []
        if debug_stats:
            print(f"[DD-dark] regions: {len(regions)}")

        for reg in regions:
            area = reg.area
            if area < AREA_MIN or area > AREA_MAX:
                continue

            per = reg.perimeter if reg.perimeter > 0 else 1.0
            circ = 4.0 * np.pi * area / (per * per)
            if circ < CIRC_MIN:
                continue

            cy, cx = reg.centroid
            cx = float(cx); cy = float(cy)
            r_est = 0.5 * reg.equivalent_diameter

            # avoid 20 hole
            d_center = np.hypot(cx - cx_board, cy - cy_board)
            if d_center < (r_center + 0.4 * r0):
                continue

            # local-annulus background
            dist = np.hypot(xx - cx, yy - cy)
            annulus = (
                (dist >= 1.3 * r_est) &
                (dist <= 2.2 * r_est) &
                play_mask &
                (~disc_mask_union)
            )
            bg_vals = lum[annulus]
            if bg_vals.size < 40:
                bg_med = board_med
            else:
                bg_med = float(np.median(bg_vals))

            # Check if disc is significantly different from local background
            diff_local = abs(reg.mean_intensity - bg_med)
            if diff_local < LOCAL_DELTA_D:
                continue

            if debug_stats:
                print(
                    f"[DD-dark-reg] ({int(cx):4d},{int(cy):4d}) "
                    f"area={area:5.0f} circ={circ:.3f} "
                    f"R~{r_est:4.1f} mean={reg.mean_intensity:.3f} "
                    f"bg={bg_med:.3f} dLoc={diff_local:.3f}"
                )

            out.append((cx, cy, r_est))

        return out

    # -------------------------------------------------------------
    # helper: extract light discs → REVISED with local contrast
    # -------------------------------------------------------------
    def extract_light_regions():
        labels = measure.label(light_mask)
        regions = measure.regionprops(labels, intensity_image=lum)
        out = []
        if debug_stats:
            print(f"[DD-light] regions: {len(regions)}")

        for reg in regions:
            area = reg.area
            if area < AREA_MIN or area > AREA_MAX:
                continue

            per = reg.perimeter if reg.perimeter > 0 else 1.0
            circ = 4.0 * np.pi * area / (per * per)
            if circ < CIRC_MIN:
                continue

            cy, cx = reg.centroid
            cx = float(cx); cy = float(cy)
            r_est = 0.5 * reg.equivalent_diameter

            d_center = np.hypot(cx - cx_board, cy - cy_board)
            if d_center < (r_center + 0.4 * r0):
                continue
            
            # NEW: Local Annulus Contrast Check
            dist = np.hypot(xx - cx, yy - cy)
            annulus = (
                (dist >= 1.3 * r_est) &
                (dist <= 2.2 * r_est) &
                play_mask &
                (~disc_mask_union)
            )
            bg_vals = lum[annulus]
            if bg_vals.size < 40:
                bg_med = board_med
            else:
                bg_med = float(np.median(bg_vals))

            # Light disc must be sufficiently brighter than its local background
            # Note: We check if mean > bg_med + threshold, since light discs are brighter
            diff_local = reg.mean_intensity - bg_med
            if diff_local < LOCAL_DELTA_L:
                continue
                
            if debug_stats:
                print(
                    f"[DD-light-reg] ({int(cx):4d},{int(cy):4d}) "
                    f"area={area:5.0f} circ={circ:.3f} R~{r_est:4.1f} "
                    f"mean={reg.mean_intensity:.3f} bg={bg_med:.3f} "
                    f"dLoc={diff_local:.3f}"
                )

            out.append((cx, cy, r_est))

        return out

    dark_cands  = extract_dark_regions()
    light_cands = extract_light_regions()

    all_cands = dark_cands + light_cands
    if not all_cands:
        print("[DD-wood] no regions survived filters.")
        return []

    # --- NMS on centres to merge duplicates from dark/light branches ---
    merged = []
    for x, y, r in all_cands:
        keep = True
        for X, Y, R in merged:
            if np.hypot(x - X, y - Y) < 0.6 * max(r, R):
                keep = False
                break
        if keep:
            merged.append((x, y, r))

    # --- colour clustering & output ---
    lab_img = color.rgb2lab(straightened_img.astype(float) / 255.0)

    feats = [
        _lab_patch_mean(
            lab_img,
            int(round(x)),
            int(round(y)),
            half=min(6, max(3, int(0.5 * r0)))
        )
        for (x, y, r) in merged
    ]
    feats = np.asarray(feats, float)
    team_labels, centroids = _kmeans2_lab(feats, seed=0)

    r_use = int(round(r0))
    results = []
    for (x, y, r), labv, team in zip(merged, feats, team_labels):
        results.append({
            "center": (int(round(x)), int(round(y))),
            "radius": r_use,
            "score":  1.0,
            "team":   int(team),
            "lab":    labv,
        })

    if cfg.get("debug_show_discs", True):
        debug_show_discs(straightened_img, results, board_result)

    return results[:max_discs]
# The _detect_discs_hough_white function is assumed to remain unchanged as it is not the focus of the current issue.
# ---------------------------------------------------------------------
# White-board Hough path (same as before)
# ---------------------------------------------------------------------
def _detect_discs_hough_white(straightened_img, board_result, config,
                              play_mask, r0, r_min, r_max,
                              r_center, max_discs, debug_stats):
    h, w = straightened_img.shape[:2]
    cx_board, cy_board = board_result['center']
    rings = board_result['rings']
    ring_radii = [float(r) for name, r in rings.items() if name != 'outer']

    THR_EDGE      = 0.06
    THR_SD_IN     = 0.20
    THR_CONTRAST  = 0.04
    THR_DELTA_BRD = 0.035
    THR_MID_STD   = 0.18
    C_LOW, C_HIGH = 0.05, 0.98

    img_f = straightened_img.astype(float) / 255.0
    sharp = filters.unsharp_mask(img_f, radius=2, amount=1.5)
    lum   = color.rgb2gray(sharp)
    lum   = exposure.equalize_adapthist(lum, clip_limit=0.01)
    med_lum = float(np.median(lum))

    e1 = feature.canny(lum,        sigma=1.5,
                       low_threshold=C_LOW, high_threshold=C_HIGH)
    e2 = feature.canny(1.0 - lum,  sigma=1.5,
                       low_threshold=C_LOW, high_threshold=C_HIGH)
    edges = np.maximum(e1, e2)

    radii = np.arange(max(2, r_min), max(3, r_max + 1), 1, dtype=int)
    hspaces = transform.hough_circle(edges, radii)
    acc, xs, ys, rs = transform.hough_circle_peaks(
        hspaces, radii, total_num_peaks=180
    )

    cands = []
    for a, x, y, r in zip(acc, xs, ys, rs):
        x = int(x); y = int(y); r = int(r)
        if not (0 <= x < w and 0 <= y < h and play_mask[y, x]):
            continue

        d_center = np.hypot(x - cx_board, y - cy_board)
        if d_center < (r_center + 0.5 * r0):
            continue

        near_ring = False
        for rr in ring_radii:
            if abs(d_center - rr) < 0.7 * r0:
                near_ring = True
                break
        if near_ring:
            continue

        e_hit = _edge_ring(edges, x, y, r, n=50)
        mu_in, sd_in, mu_out = _inside_stats(lum, x, y, r)
        contrast    = abs(mu_in - mu_out)
        delta_board = abs(mu_in - med_lum)
        mid_std, _  = _angular_uniformity(lum, x, y, r, n=40, frac=0.5)

        if (e_hit      >= THR_EDGE and
            sd_in      <= THR_SD_IN and
            contrast   >= THR_CONTRAST and
            delta_board >= THR_DELTA_BRD and
            mid_std    <= THR_MID_STD):
            if debug_stats:
                print(
                    f"[DD-white] ({x:4d},{y:4d}) r={r:2d} "
                    f"e={e_hit:.3f} sd_in={sd_in:.3f} "
                    f"contr={contrast:.3f} dB={delta_board:.3f} "
                    f"mid_std={mid_std:.3f}"
                )
            cands.append((x, y, r, float(e_hit)))

    if not cands:
        print("[DD-white] No disc candidates passed thresholds.")
        return []

    cands.sort(key=lambda t: t[3], reverse=True)
    picked = []
    for x, y, r, s in cands:
        keep = True
        for X, Y, R, S in picked:
            if np.hypot(x - X, y - Y) < 0.9 * max(r, R):
                keep = False
                break
        if keep:
            picked.append((x, y, r, s))

    if not picked:
        print("[DD-white] All candidates suppressed by NMS.")
        return []

    lab_img = color.rgb2lab(straightened_img.astype(float) / 255.0)
    feats = [
        _lab_patch_mean(lab_img, x, y,
                        half=min(6, max(3, int(0.5 * r0))))
        for (x, y, r, s) in picked
    ]
    team_labels, centroids = _kmeans2_lab(feats, seed=0)

    r_use = int(round(r0))
    results = []
    for (x, y, r, s), labv, team in zip(picked, feats, team_labels):
        results.append({
            'center': (int(x), int(y)),
            'radius': r_use,
            'score': float(s),
            'team':  int(team),
            'lab':   labv,
        })

    if config.get('disc_detection', {}).get("debug_show_discs", True):
        debug_show_discs(straightened_img, results, board_result)

    return results[:max_discs]



def debug_show_discs(image, discs, board_result=None, title="Detected discs"):
    """
    Quick visualization of disc detections.
    Draws a circle for each disc and labels it with its index.
    Colours indicate team 0 vs 1 if available.
    """
    if image is None or len(discs) == 0:
        print("[DD] No discs to visualize")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    # Show board centre and rings
    if board_result is not None:
        cx, cy = board_result['center']
        ax.plot(cx, cy, 'rx', markersize=8, mew=2)
        for name, r in board_result['rings'].items():
            circ = patches.Circle(
                (cx, cy), r,
                linewidth=1.0,
                edgecolor='white',
                facecolor='none',
                alpha=0.4
            )
            ax.add_patch(circ)

    for i, d in enumerate(discs):
        x, y = d['center']
        r = d['radius']
        team = d.get('team', 0)
        edge_col = 'cyan' if team == 0 else 'magenta'

        circ = patches.Circle(
            (x, y), r,
            linewidth=2.0,
            edgecolor=edge_col,
            facecolor='none'
        )
        ax.add_patch(circ)
        ax.text(
            x, y, str(i),
            color='yellow',
            fontsize=9,
            ha='center', va='center',
            weight='bold'
        )

    ax.set_title(f"{title} (N={len(discs)})")
    ax.axis('off')
    plt.show()



# -----------------------------
# STEP 6: Colour grouping
# -----------------------------
def extract_disc_colours(image, discs, patch_half=3):
    """
    Returns Nx3 Lab features (mean Lab in a (2*patch_half+1)^2 square).
    """
    if len(discs) == 0:
        return np.zeros((0, 3), dtype=float)
    lab = color.rgb2lab(image)
    h, w = lab.shape[:2]
    feats = []
    for d in discs:
        x, y = d['center']
        x0 = max(0, x - patch_half); x1 = min(w, x + patch_half + 1)
        y0 = max(0, y - patch_half); y1 = min(h, y + patch_half + 1)
        patch = lab[y0:y1, x0:x1, :]
        feats.append(np.mean(patch.reshape(-1, 3), axis=0))
    return np.array(feats, dtype=float)


def cluster_teams_lab(features, n_clusters=2, n_init=5, seed=0):
    """
    Simple k-means in Lab using Lloyd's algorithm (no sklearn dependency).
    Returns labels (0/1) and centroids (2x3).
    """
    if features.shape[0] == 0:
        return np.array([], dtype=int), np.zeros((n_clusters, 3))
    rng = np.random.default_rng(seed)
    # k-means++ init
    centroids = [features[rng.integers(0, len(features))]]
    for _ in range(1, n_clusters):
        d2 = np.min([np.sum((features - c) ** 2, axis=1) for c in centroids], axis=0)
        probs = d2 / (d2.sum() + 1e-9)
        centroids.append(features[rng.choice(len(features), p=probs)])
    centroids = np.stack(centroids, axis=0)

    for _ in range(30):
        # assign
        d = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d, axis=1)
        # update
        new_centroids = []
        for k in range(n_clusters):
            if np.any(labels == k):
                new_centroids.append(features[labels == k].mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.stack(new_centroids, axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids
    return labels.astype(int), centroids


def deltaE_ab(c1, c2):
    """Euclidean distance in ab-plane (ignore L for robustness)."""
    return float(np.linalg.norm(c1[1:] - c2[1:]))


def check_colour_similarity(features, labels):
    """
    Returns (similarity_score, uncertain_indices).
    similarity_score is ΔE between cluster centroids in ab-plane.
    """
    if len(features) == 0:
        return 0.0, []

    labs0 = features[labels == 0]
    labs1 = features[labels == 1]
    if len(labs0) == 0 or len(labs1) == 0:
        return 0.0, list(range(len(features)))

    c0 = labs0.mean(axis=0)
    c1 = labs1.mean(axis=0)
    score = deltaE_ab(c0, c1)

    # simple per-point confidence: distance to own centroid vs other
    uncertain = []
    for i, f in enumerate(features):
        d0 = np.linalg.norm(f - c0)
        d1 = np.linalg.norm(f - c1)
        if abs(d0 - d1) < 3.0:  # small margin
            uncertain.append(i)
    return score, uncertain


# -----------------------------
# STEP 7: Scoring per disc
# -----------------------------
@dataclass
class DiscScore:
    idx: int
    score: int
    confidence: float
    flags: list  # list of strings


def calculate_disc_scores(discs, mask, line_band, board_info):
    """
    For each disc, sample perimeter points, apply line rule, and center-20 check.
    Returns list of DiscScore.
    """
    results = []
    if len(discs) == 0:
        return results

    h, w = mask.shape
    cx, cy = board_info['center']
    r_center = board_info['radii']['center']
    r_outer = board_info['radii']['outer']

    for i, d in enumerate(discs):
        x, y = d['center']
        r = d['radius']
        flags = []
        # perimeter sampling
        n_samples = 32
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        vals = []
        lower_touch = False
        for ang in angles:
            xs = int(round(x + r * np.cos(ang)))
            ys = int(round(y + r * np.sin(ang)))
            if xs < 0 or xs >= w or ys < 0 or ys >= h:
                vals.append(0)
                lower_touch = True
                continue
            vals.append(mask[ys, xs])
            if line_band[ys, xs]:
                lower_touch = True

        # majority region
        if len(vals) == 0:
            results.append(DiscScore(i, 0, 0.0, ['no_samples']))
            continue
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        maj = max(counts.items(), key=lambda kv: kv[1])[0]

        # center-20 proxy: fully inside hole
        center_dist = np.hypot(x - cx, y - cy)
        fully_in_20 = (center_dist + r) < (r_center - 1.0)

        base_score = maj
        if maj == 15 and fully_in_20:
            base_score = 20

        # line rule: if touches line band or any sample in lower region, take lower
        if lower_touch:
            # Lower neighbor of base_score (20->15, 15->10, 10->5, 5->0)
            lower_map = {20: 15, 15: 10, 10: 5, 5: 0, 0: 0}
            base_score = lower_map[base_score]
            flags.append('line_touch')

        # sanity checks
        if (center_dist + r) > (r_outer + 1):
            base_score = 0
            flags.append('outside')

        # confidence: proportion of perimeter agreeing with base_score
        agree = np.mean([v == maj for v in vals])
        conf = float(agree) if 'line_touch' not in flags else max(0.3, float(agree) * 0.8)

        results.append(DiscScore(i, int(base_score), conf, flags))
    return results


def calculate_team_totals(disc_scores, team_labels):
    """
    Sums 5/10/15 visible scores per team.
    20s will be added separately from user input.
    """
    t0 = 0
    t1 = 0
    for ds in disc_scores:
        if ds.score == 20:
            # do not add; handled via user input because the disc is removed
            continue
        if team_labels[ds.idx] == 0:
            t0 += ds.score
        else:
            t1 += ds.score
    return t0, t1


# -----------------------------
# STEP 8/9: Overlay and utils
# -----------------------------
def create_results_overlay(image, board_result, discs, labels, disc_scores):
    """
    Draw rings, discs with team color edges, and per-disc scores.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    cx, cy = board_result['center']
    for name, r in board_result['rings'].items():
        circ = patches.Circle((cx, cy), r, linewidth=1.5, edgecolor='white', facecolor='none', alpha=0.7)
        ax.add_patch(circ)
        ax.text(cx + r + 4, cy, name, color='white', fontsize=8)

    for ds in disc_scores:
        x, y = discs[ds.idx]['center']
        r = discs[ds.idx]['radius']
        team = labels[ds.idx]
        edge = 'tab:blue' if team == 0 else 'tab:orange'
        c = patches.Circle((x, y), r, linewidth=2.0, edgecolor=edge, facecolor='none')
        ax.add_patch(c)
        ax.text(x, y, str(ds.score), color='white', ha='center', va='center', fontsize=9, weight='bold')
        if len(ds.flags) > 0:
            ax.plot([x], [y - r - 6], marker='v', markersize=6, color='red')

    ax.axis('off')
    fig.canvas.draw()
    overlay = np.asarray(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return overlay


def calculate_tournament_points(team1_total, team2_total):
    """
    Simple round points: 2–0 to winner, 1–1 if tie.
    """
    if team1_total > team2_total:
        return "2-0"
    elif team2_total > team1_total:
        return "0-2"
    else:
        return "1-1"


# -----------------------------
# STEP 10: Evaluation
# -----------------------------
def calculate_detection_score(detected_xy, gt_xy, match_threshold=5.0):
    """
    detected_xy: [(x,y), ...]
    gt_xy: [(x,y), ...]
    """
    if len(gt_xy) == 0 and len(detected_xy) == 0:
        return 1.0, 0, 0, 0
    if len(gt_xy) == 0:
        return 0.0, 0, 0, len(detected_xy)

    gt_used = np.zeros(len(gt_xy), dtype=bool)
    correct = 0
    for dx, dy in detected_xy:
        dists = [np.hypot(dx - gx, dy - gy) if not gt_used[j] else 1e9 for j, (gx, gy) in enumerate(gt_xy)]
        jmin = int(np.argmin(dists))
        if dists[jmin] <= match_threshold:
            correct += 1
            gt_used[jmin] = True
    missed = int(np.sum(~gt_used))
    extras = len(detected_xy) - correct
    score = (2 * correct) / (2 * correct + missed + extras + 1e-9)
    return float(score), int(correct), int(missed), int(extras)


def evaluate_ring_accuracy(pred_scores, gt_scores, near_line_flags=None):
    """
    pred_scores: [5/10/15/20/0 per disc] aligned with GT
    gt_scores:   same length
    near_line_flags: optional boolean list of same length
    """
    assert len(pred_scores) == len(gt_scores)
    n = len(pred_scores)
    overall = np.mean([int(p == g) for p, g in zip(pred_scores, gt_scores)]) if n else 0.0
    if near_line_flags is not None and len(near_line_flags) == n and np.any(near_line_flags):
        nl_idx = [i for i, v in enumerate(near_line_flags) if v]
        nl_acc = np.mean([int(pred_scores[i] == gt_scores[i]) for i in nl_idx])
    else:
        nl_acc = 0.0
    return float(overall), float(nl_acc)


def evaluate_final_scores(pred_totals, gt_totals):
    """
    pred_totals: (t1, t2)
    gt_totals:   (t1, t2)
    """
    t1p, t2p = pred_totals
    t1g, t2g = gt_totals
    exact = int((t1p == t1g) and (t2p == t2g))
    err = abs(t1p - t1g) + abs(t2p - t2g)
    return float(exact), float(err)


