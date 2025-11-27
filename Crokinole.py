"""
Crokinole Score Detection System
Core functions for board detection, disc detection, and scoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color, feature, transform, draw, filters, measure, morphology, exposure, segmentation
from matplotlib import patches
from dataclasses import dataclass


def preprocess_image(img, target_size=None, max_dimension=1200):
    """Resize and normalize image to RGB uint8 format."""
    if len(img.shape) == 2:
        img = color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = color.rgba2rgb(img)
    
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    h, w = img.shape[:2]
    
    if target_size is not None:
        if (h, w) != target_size:
            img = transform.resize(img, target_size, anti_aliasing=True, preserve_range=True).astype(np.uint8)
    elif max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = transform.resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(np.uint8)
    
    return img

def adjust_config_for_image(img, config):
    """
    Placeholder function for dynamic config adjustment.
    Step size is now statically configured in config.py.
    """
    # No dynamic adjustments - step_size is static at 6
    return config


def display_image(img, title="Image", figsize=(8, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def display_images(images, titles=None, figsize=(14, 7)):
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



def detect_edges(img, config):
    """Detect edges using Canny edge detection."""
    board_cfg = config['board_detection']
    
    if img.ndim == 2:
        gray_image = img
    else:
        if img.shape[-1] == 4:
            img_rgb = color.rgba2rgb(img)
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
    """Detect board and rings using ring_5 as primary reference."""
    h, w = edges.shape[:2]
    board_cfg = config['board_detection']
    ring_ratios = config['ring_ratios']
    ring_search_cfg = config['ring_search']
    verbose = config.get('verbose', False)

    min_radius = int(min(h, w) * board_cfg['min_circle_ratio'])
    max_radius = int(min(h, w) * board_cfg['max_circle_ratio'])
    min_inner = int(min_radius * 0.05)
    
    # Adaptive step size: scale with image dimensions to keep search space manageable
    # Target ~200 radii regardless of image size for consistent performance
    base_step = ring_search_cfg['step_size']
    img_dim = min(h, w)
    radius_range = max_radius - min_inner
    
    # Calculate adaptive step to maintain ~200 search radii
    target_num_radii = 200
    adaptive_step = max(base_step, int(radius_range / target_num_radii))
    
    all_radii = np.arange(min_inner, max_radius, adaptive_step)

    if verbose:
        print(f"Image dimensions: {h}x{w}, Radius range: {min_inner}-{max_radius}")
        print(f"Adaptive step size: {adaptive_step} (base: {base_step})")
        print(f"Running single Hough transform for {len(all_radii)} radii...")
    hough_res = transform.hough_circle(edges, all_radii)
    accums, cx, cy, radii_detected = transform.hough_circle_peaks(
        hough_res, all_radii, total_num_peaks=120
    )

    if len(cx) == 0:
        if verbose:
            print("No circles found")
        return None

    all_circles = []
    for x, y, r, acc in zip(cx, cy, radii_detected, accums):
        all_circles.append({
            "center": (int(x), int(y)),
            "radius": int(r),
            "acc": float(acc),
        })

    ring5_min = int(min_radius * 0.95)
    ring5_candidates = [c for c in all_circles if c['radius'] >= ring5_min]
    ring5_candidates.sort(key=lambda c: c['radius'], reverse=True)

    if verbose:
        print(f"Found {len(all_circles)} circles total")
        print(f"Ring5 minimum radius: {ring5_min} (min_radius={min_radius}, max_radius={max_radius})")
        print(f"Circle radius range: {min(c['radius'] for c in all_circles) if all_circles else 0} to {max(c['radius'] for c in all_circles) if all_circles else 0}")
        print(f"Ring5 candidates (radius >= {ring5_min}): {len(ring5_candidates)}")
        if ring5_candidates:
            print(f"Top 5 ring5 candidates: {[c['radius'] for c in ring5_candidates[:5]]}")

    if not ring5_candidates:
        if verbose:
            print("No suitable ring_5 candidates")
        return None

    for ring5 in ring5_candidates[:5]:
        ring5_center = ring5['center']
        ring5_radius = float(ring5['radius'])

        board_radius = int(ring5_radius / ring_ratios['ring_5'])

        max_offset = board_radius * ring_search_cfg['max_center_offset']

        detected_rings = {
            "ring_5": {
                'radius': ring5['radius'],
                'center': ring5_center
            }
        }
        used_indices = set()

        inner_ratios = {k: v for k, v in ring_ratios.items() if k not in ['ring_5', 'center']}
        sorted_rings = sorted(inner_ratios.items(), key=lambda kv: kv[1], reverse=True)

        for ring_name, ratio in sorted_rings:
            expected_r = board_radius * ratio
            radius_tol = expected_r * 0.15
            if radius_tol < 5:
                radius_tol = 5
            
            center_tol = board_radius * 0.03
            if center_tol < 5:
                center_tol = 5

            best_idx = None
            best_match = None
            best_score = float('inf')

            for idx, circle in enumerate(all_circles):
                if idx in used_indices:
                    continue

                r_meas = circle['radius']
                # Check radius is within tolerance
                radius_diff = abs(r_meas - expected_r)
                if radius_diff > radius_tol:
                    continue

                # Check center proximity to ring_5 center (PRIMARY constraint)
                cx_c, cy_c = circle['center']
                dx = cx_c - ring5_center[0]
                dy = cy_c - ring5_center[1]
                center_dist = np.hypot(dx, dy)
                if center_dist > center_tol:
                    continue

                radius_score = radius_diff / radius_tol
                center_score = center_dist / center_tol
                center_weight = 1.0 / max(ratio, 0.05)
                score = center_score * center_weight + radius_score
                
                if score < best_score:
                    best_score = score
                    best_match = circle
                    best_idx = idx

            if best_match is None:
                if verbose:
                    print(f"    Failed to find {ring_name} (expected r={expected_r:.1f}, tolerance=±{radius_tol:.1f}, center_tol={center_tol:.1f}), trying next ring_5 candidate")
                detected_rings = None
                break

            detected_rings[ring_name] = {
                'radius': int(best_match['radius']),
                'center': best_match['center']
            }
            if verbose:
                print(f"    Found {ring_name}: radius={best_match['radius']}, center={best_match['center']}")
            used_indices.add(best_idx)

        if detected_rings is None:
            continue

        if len(detected_rings) != 3:
            continue

        xs = [ring_data['center'][0] for ring_data in detected_rings.values()]
        ys = [ring_data['center'][1] for ring_data in detected_rings.values()]
        cx_ref = int(round(np.mean(xs)))
        cy_ref = int(round(np.mean(ys)))
        board_center = (cx_ref, cy_ref)
        
        center_radius = int(round(board_radius * ring_ratios['center']))
        
        detected_rings['center'] = {
            'radius': center_radius,
            'center': board_center
        }
        if verbose:
            print(f"    Calculated center: radius={center_radius}, center={board_center}")

        if verbose:
            print(
                "Board found: centre={} outer_radius≈{} (from ring_5={})".format(
                    board_center, board_radius, ring5_radius
                )
            )
            print("  Detected rings:")
            for ring_name, ring_data in detected_rings.items():
                print(f"    {ring_name}: r={ring_data['radius']}, center={ring_data['center']}")

        return {
            "center": board_center,
            "radius": board_radius,
            "rings": detected_rings
        }

    if verbose:
        print("No valid board pattern found after trying ring_5 candidates")
    return None



def visualize_board_detection(img, board_result):
    """Visualize detected board and rings."""
    if board_result is None:
        return
    
    board_center = board_result['center']
    detected_rings = board_result['rings']
    
    colors = {
        'ring_5': [255, 0, 0],      # Red
        'ring_10': [255, 165, 0],   # Orange
        'ring_15': [255, 255, 0],   # Yellow
        'center': [0, 0, 255]       # Blue
    }
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    overlay = img.copy()
    
    # Draw each ring at its actual detected center and radius
    for ring_name, ring_data in detected_rings.items():
        if ring_name in colors:
            ring_center = ring_data['center']
            ring_radius = ring_data['radius']
            
            circy, circx = draw.circle_perimeter(
                ring_center[1], ring_center[0], ring_radius,
                shape=img.shape[:2]
            )
            overlay[circy, circx] = colors[ring_name]
    
    ax.imshow(overlay)
    # Mark the refined board center (average of all ring centers)
    ax.plot(board_center[0], board_center[1], 'ro', markersize=8)
    ax.set_title("Detected Board and Rings")
    ax.axis('off')
    plt.show()



def create_scoring_regions(img_shape, board_result):
    """
    Create a segmentation mask with scoring regions based on detected rings.
    Each ring has its own center and radius from detection.
    
    Ring naming (from outer to inner):
    - ring_5: outermost ring (boundary between 0pt outside and 5pt region)
    - ring_10: second ring (boundary between 5pt and 10pt regions)
    - ring_15: third ring (boundary between 10pt and 15pt regions)
    - center: innermost circle (boundary for 20pt center hole)
    
    Returns a mask where pixel values represent point values:
    0 = outside board (beyond ring_5)
    5 = between ring_5 and ring_10
    10 = between ring_10 and ring_15
    15 = between ring_15 and center
    20 = inside center hole
    """
    if board_result is None:
        return None
    
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    detected_rings = board_result['rings']
    
    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    
    # Calculate distance from each ring's individual center
    ring_5_data = detected_rings.get('ring_5')
    ring_10_data = detected_rings.get('ring_10')
    ring_15_data = detected_rings.get('ring_15')
    center_data = detected_rings.get('center')
    
    # 5 point region: inside ring_5 but outside ring_10
    if ring_5_data and ring_10_data:
        dist_to_ring5 = np.sqrt((x - ring_5_data['center'][0])**2 + (y - ring_5_data['center'][1])**2)
        dist_to_ring10 = np.sqrt((x - ring_10_data['center'][0])**2 + (y - ring_10_data['center'][1])**2)
        mask[(dist_to_ring5 <= ring_5_data['radius']) & (dist_to_ring10 > ring_10_data['radius'])] = 5
    
    # 10 point region: inside ring_10 but outside ring_15
    if ring_10_data and ring_15_data:
        dist_to_ring10 = np.sqrt((x - ring_10_data['center'][0])**2 + (y - ring_10_data['center'][1])**2)
        dist_to_ring15 = np.sqrt((x - ring_15_data['center'][0])**2 + (y - ring_15_data['center'][1])**2)
        mask[(dist_to_ring10 <= ring_10_data['radius']) & (dist_to_ring15 > ring_15_data['radius'])] = 10
    
    # 15 point region: inside ring_15 but outside center
    if ring_15_data and center_data:
        dist_to_ring15 = np.sqrt((x - ring_15_data['center'][0])**2 + (y - ring_15_data['center'][1])**2)
        dist_to_center = np.sqrt((x - center_data['center'][0])**2 + (y - center_data['center'][1])**2)
        mask[(dist_to_ring15 <= ring_15_data['radius']) & (dist_to_center > center_data['radius'])] = 15
    
    # 20 point region: inside center hole
    if center_data:
        dist_to_center = np.sqrt((x - center_data['center'][0])**2 + (y - center_data['center'][1])**2)
        mask[dist_to_center <= center_data['radius']] = 20
    
    return mask


def visualize_scoring_regions(img, scoring_mask):
    """
    Visualize the scoring regions with colored overlay.
    Each scoring region gets a distinct color.
    """
    if scoring_mask is None:
        print("No scoring mask to visualize")
        return
    
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
    
    alpha = 0.5
    if len(img.shape) == 3:
        blended = (alpha * img + (1 - alpha) * overlay).astype(np.uint8)
    else:
        img_rgb = np.stack([img, img, img], axis=2)
        blended = (alpha * img_rgb + (1 - alpha) * overlay).astype(np.uint8)
    
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title("Scoring Regions Mask")
    axes[1].axis('off')
    
    axes[2].imshow(blended)
    axes[2].set_title("Blended Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    unique_scores = np.unique(scoring_mask)
    print(f"Scoring regions: {sorted(unique_scores)} points")
    for score in sorted(unique_scores):
        count = np.sum(scoring_mask == score)
        percentage = (count / scoring_mask.size) * 100
        print(f"  {score}pt region: {count} pixels ({percentage:.1f}%)")
# -----------------------------
# STEP 5: Disc detection + colour grouping
# -----------------------------

def _board_brightness(straightened_img, board_result, inset_px=8):
    """
    Estimate board 'whiteness' / brightness in the playable area.

    Returns a single scalar in [0, 1] (median grayscale value inside play area).
    We use the *raw* grayscale (no equalize_adapthist) so numbers are comparable
    across runs, e.g. 0.53, 0.94, etc.
    """
    h, w = straightened_img.shape[:2]

    # raw grayscale in [0,1]
    img_f = straightened_img.astype(float) / 255.0
    gray  = color.rgb2gray(img_f)

    # playable mask (uses ring_5 / outer)
    play_mask = _play_area_mask(
        straightened_img.shape,
        board_result,
        inset_px=inset_px
    )

    vals = gray[play_mask]
    if vals.size == 0:
        # fallback: whole image
        vals = gray.ravel()

    return float(np.median(vals))


def _expected_disc_radius(board_result, config, pad_frac=0.10, min_px=4):
    """
    Use the 'center' ring if present; otherwise approximate it from outer radius
    and CONFIG['ring_ratios']['center'].
    """
    rings = board_result['rings']
    r0 = _ring_radius_value(rings.get('center', 0.0), default=0.0)

    # Fallback: approximate from outer radius + ratio
    if r0 <= 0.0:
        outer_r = _ring_radius_value(rings.get('outer', 0.0), default=0.0)
        if outer_r > 0:
            center_ratio = config.get('ring_ratios', {}).get('center', 0.05)
            r0 = outer_r * center_ratio

    if r0 <= 0.0:
        return None, None, None

    band = max(min_px, r0 * pad_frac)
    return r0, max(2, int(r0 - band)), int(r0 + band)


def _play_area_mask(shape, board_result, inset_px=4):
    """
    Mask for the playable wood:
    inside ring_5 (if present) or 0.95*outer as fallback.
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = board_result['center']
    rings = board_result.get('rings', {})

    rout = _ring_radius_value(rings.get('outer', 0.0), default=0.0)

    r5_default = 0.95 * rout if rout > 0 else 0.0
    r5 = _ring_radius_value(rings.get('ring_5', r5_default), default=r5_default)


    r_play_outer = r5 if r5 > 0 else rout
    if rout > 0 and r5 > 0:
        r_play_outer = min(r5, rout)

    r_play_outer = max(0.0, r_play_outer - inset_px)
    dist = np.hypot(xx - cx, yy - cy)
    return dist <= r_play_outer


def _edge_ring(edg, x, y, r, n=48):
    """Mean edge strength along a circle of radius r around (x,y)."""
    h, w = edg.shape
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.clip((x + r * np.cos(ang)).round().astype(int), 0, w - 1)
    ys = np.clip((y + r * np.sin(ang)).round().astype(int), 0, h - 1)
    return float(edg[ys, xs].mean())


def _inside_stats(lum, x, y, r):
    """
    Mean/std inside disc, and mean in a thin outer ring.
    Returns (mu_in, sd_in, mu_out).
    """
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


def _angular_uniformity(lum, x, y, r, n=36, frac=0.5):
    """
    Sample grayscale values on a mid-radius circle (frac * r) around (x,y).
    Returns (std, mean). Real discs should have relatively low std.
    """
    h, w = lum.shape
    if r <= 2:
        return 0.0, 0.0

    radius = max(2.0, frac * float(r))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.clip((x + radius * np.cos(angles)).round().astype(int), 0, w - 1)
    ys = np.clip((y + radius * np.sin(angles)).round().astype(int), 0, h - 1)
    vals = lum[ys, xs]
    return float(vals.std()), float(vals.mean())


def _lab_patch_mean(lab_img, x, y, half=3):
    """Mean Lab colour in a small square patch around (x,y)."""
    h, w = lab_img.shape[:2]
    x0 = max(0, x - half)
    x1 = min(w, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)
    patch = lab_img[y0:y1, x0:x1, :]
    return np.mean(patch.reshape(-1, 3), axis=0)


def _kmeans2_lab(feats, seed=0):
    """
    Tiny K=2 k-means on Lab features; returns labels (0/1) and centroids (2x3).

    Handles edge cases:
      - 0 discs  → returns empty labels, dummy centroids
      - 1 disc   → puts it in cluster 0 and duplicates centroid for cluster 1
    """
    # No discs
    if len(feats) == 0:
        return np.array([], dtype=int), np.zeros((2, 3))

    X = np.asarray(feats, float)
    n = X.shape[0]

    # Single disc: avoid k-means++ with degenerate probabilities
    if n == 1:
        labels = np.array([0], dtype=int)
        c0 = X[0]
        C = np.stack([c0, c0], axis=0)   # both centroids identical
        return labels, C

    rng = np.random.default_rng(seed)

    # k-means++ init for K=2
    c0 = X[rng.integers(0, n)]
    d2 = np.sum((X - c0) ** 2, axis=1)
    total = d2.sum()

    # Just in case all points are identical (total == 0)
    if total <= 0:
        labels = np.zeros(n, dtype=int)
        c0 = X.mean(axis=0)
        C = np.stack([c0, c0], axis=0)
        return labels, C

    probs = d2 / total
    c1 = X[rng.choice(n, p=probs)]
    C = np.stack([c0, c1], axis=0)

    for _ in range(20):
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        L = np.argmin(D, axis=1)
        C_new = []
        for k in (0, 1):

            if np.any(L == k):
                C_new.append(X[L == k].mean(axis=0))
            else:
                C_new.append(C[k])
        C_new = np.stack(C_new, axis=0)
        if np.allclose(C_new, C, atol=1e-4):
            break
        C = C_new

    return L.astype(int), C

def _select_disc_params(board_brightness, r0):
    """
    Choose disc-detection thresholds based on board brightness.

    board_brightness is in [0,1], e.g. ~0.98 for very bright white board,
    ~0.23 for your dark blue board.
    """
    # --- PRESET 1: Bright white board (your white test image) -----------
    if 0.95 <= board_brightness <= 1.01:
        return {
            "name": "bright_white",
            "ring_margin": 0.25 * r0,   # suppress only very near rings (not center)
            "e_hit_min": 0.28,
            "sd_in_max": 0.15,
            "contrast_min": 0.18,
            "delta_board_min": 0.1,
            "mid_std_max": 0.80,
        }

    # --- PRESET 2: Medium wood boards -----------------------------------
    if 0.45 <= board_brightness < 0.95:
        # Slightly darker / higher contrast wood
        if board_brightness < 0.65:
            return {
                "name": "mid_wood",
                "ring_margin": 0.30 * r0,
                # slightly easier edge requirement
                "e_hit_min": 0.18,      # was 0.22

                # allow a bit more variation inside the disc
                "sd_in_max": 0.25,      # was 0.18

                # keep brightness-based filters as-is
                "contrast_min": 0.08,
                "delta_board_min": 0.09,

                # allow slightly less perfect angular uniformity
                "mid_std_max": 1.00,    # was 0.90
            }
        if 0.65 <= board_brightness <0.8 :
            return{
                "name": "light_wood_hard",
                "ring_margin": 0.06 * r0,   # let discs a bit closer to rings
                "e_hit_min": 0.25,          # allow weaker edges
                "sd_in_max": 0.22,          # allow more texture / highlights inside
                "contrast_min": 0.01,       # OK with low contrast vs surroundings
                "delta_board_min": 0.08,    # OK with being close to global board L
                "mid_std_max": 1.0,   
            }
        if 0.8 <= board_brightness <0.95 :
            return{
                "name": "light_wood_hard",
                "ring_margin": 0.32 * r0,   # let discs a bit closer to rings
                "e_hit_min": 0.14,          # allow weaker edges
                "sd_in_max": 0.22,          # allow more texture / highlights inside
                "contrast_min": 0.001,       # OK with low contrast vs surroundings
                "delta_board_min": 0.001,    # OK with being close to global board L
                "mid_std_max": 1.3,   
            }

        # Brighter mid-wood (like your new board: ~0.688)
        # → make thresholds more forgiving for pale / white discs.
        return {
            "name": "mid_wood_soft",
            "ring_margin": 0.32 * r0,   # let discs a bit closer to rings
            "e_hit_min": 0.10,          # allow weaker edges
            "sd_in_max": 0.25,          # allow more texture / highlights inside
            "contrast_min": 0.001,       # OK with low contrast vs surroundings
            "delta_board_min": 0.001,    # OK with being close to global board L
            "mid_std_max": 1.3,        # don’t over-penalize mild angular texture
        }

    # --- PRESET 3: Dark board / odd lighting (your blue board) ----------
    if board_brightness < 0.40:
        # Dark cloth / shaded board.
        return {
            "name": "dark_or_shaded",
            "ring_margin": 0.40 * r0,
            "e_hit_min": 0.14,
            "sd_in_max": 0.36,
            "contrast_min": 0.01,
            "delta_board_min": 0.01,
            "mid_std_max": 1.70,
        }
    if 0.40 <= board_brightness < 0.47:
        return {
            "name": "dark_soft",
            "ring_margin": 0.30 * r0,  # keep same ring margin for now
            # MAIN loosened thresholds:
            "e_hit_min":    0.22,   # was 0.25 – accept slightly weaker edges
            "sd_in_max":    0.23,   # was 0.17 – allow a bit more texture
            "contrast_min": 0.07,   # was 0.10 – accept slightly lower contrast
            "delta_board_min": 0.05, # was 0.07 – disc can be closer to board L
            "mid_std_max":  1.05,   # was 0.90 – allow more angular variation
    }

    # --- Fallback: if we somehow land between bands ----------------------
    return {
        "name": "default",
        "ring_margin": 0.30 * r0,
        "e_hit_min": 0.25,
        "sd_in_max": 0.17,
        "contrast_min": 0.10,
        "delta_board_min": 0.07,
        "mid_std_max": 0.90,
    }

def _ring_radius_value(v, default=0.0):
    """
    Extract a numeric radius from a ring entry.

    Handles cases where v is:
      - a plain number (int/float)
      - a dict with 'radius' or 'r'
      - a (x, y, r) style tuple/list
    Falls back to `default` if it can't make sense of it.
    """
    if isinstance(v, (int, float, np.floating)):
        return float(v)

    if isinstance(v, dict):
        for key in ("radius", "r", "rad"):
            val = v.get(key, None)
            if isinstance(val, (int, float, np.floating)):
                return float(val)
        # last resort: try to cast whole dict, but if it fails, use default
        try:
            return float(v)
        except Exception:
            return float(default)

    if isinstance(v, (tuple, list)) and len(v) >= 3:
        r = v[2]
        if isinstance(r, (int, float, np.floating)):
            return float(r)

    try:
        return float(v)
    except Exception:
        return float(default)

def detect_discs(straightened_img, board_result, config):
    """
    Dispatch between:
      - Smart Ring logic for the mid-wood board (~0.5 brightness).
      - Generic brightness-preset logic (_select_disc_params) for everything else.
    """
    # Use the same brightness definition you logged before
    board_bright = _board_brightness(straightened_img, board_result, inset_px=8)

    # Mid-wood band where the Smart Ring algo worked well (your 0.518 board)
    if 0.45 <= board_bright < 0.60:
        print(f"[DD] board_brightness={board_bright:.3f} → Smart Ring (mid-wood) path")
        return _detect_discs_midwood_smart(straightened_img, board_result, config)
    else:
        print(f"[DD] board_brightness={board_bright:.3f} → generic preset path")
        return _detect_discs_generic(straightened_img, board_result, config, board_bright)

def _detect_discs_generic(straightened_img, board_result, config, board_bright=None):
    """
    Original disc detection tied to _select_disc_params.
    Used for:
      - bright white board
      - bright mid-wood board (~0.68)
      - dark/shaded board
      - any 'other' brightness
    """
    h, w = straightened_img.shape[:2]
    r0, r_min, r_max = _expected_disc_radius(
        board_result, config, pad_frac=0.30
    )

    if r0 is None:
        print("[DD] No center ring radius; cannot infer disc size.")
        return []

    # ---- Board centre & ring radii for geometric filtering ------------------
    bcx, bcy = board_result['center']
    raw_rings = board_result.get('rings', {})
    ring_radii = []
    for name, rinfo in raw_rings.items():
        if name == 'outer':
            continue
        ring_radii.append(_ring_radius_value(rinfo, default=0.0))

    cfg = config.get('disc_detection', {})
    max_discs = cfg.get('max_discs', 28)

    # ---- Estimate board brightness and choose thresholds --------------------
    if board_bright is None:
        board_bright = _board_brightness(straightened_img, board_result, inset_px=8)

    params = _select_disc_params(board_bright, r0)
    preset_name = params.get("name", "default")
    print(f"[DD] preset={preset_name}")
    white_like = (preset_name in ("bright_white", "mid_wood_soft"))

    if preset_name == "bright_white":
        # white board: allow a disc slightly off-center in the 20-hole
        center_forbid = 0.5 * r0
    else:
        # wood boards: just kill any circle whose centre lives in the 20-hole
        center_forbid = 0.9 * r0

    ring_margin      = params["ring_margin"]
    e_hit_min        = params["e_hit_min"]
    sd_in_max        = params["sd_in_max"]
    contrast_min     = params["contrast_min"]
    delta_board_min  = params["delta_board_min"]
    mid_std_max      = params["mid_std_max"]

    # --- Pre-processing ------------------------------------------------------
    img_f = straightened_img.astype(float) / 255.0
    sharp = filters.unsharp_mask(img_f, radius=2, amount=1.5)
    lum   = color.rgb2gray(sharp)
    lum   = exposure.equalize_adapthist(lum, clip_limit=0.01)

    med = float(np.median(lum))
    low_t, high_t = max(0.05, 0.66 * med), min(0.98, 1.33 * med)

    e1 = feature.canny(
        lum, sigma=1.5,
        low_threshold=low_t, high_threshold=high_t
    )
    e2 = feature.canny(
        1.0 - lum, sigma=1.5,
        low_threshold=low_t, high_threshold=high_t
    )
    edges = np.maximum(e1, e2)

    # --- Restrict to playable wood ------------------------------------------
    play_mask = _play_area_mask(
        straightened_img.shape,
        board_result,
        inset_px=max(3, int(0.003 * min(h, w)))
    )

    # --- Hough circles in tight radius band ---------------------------------
    radii = np.arange(max(2, r_min), max(3, r_max + 1), 1, dtype=int)
    hspaces = transform.hough_circle(edges, radii)
    acc, hcx, hcy, rs = transform.hough_circle_peaks(
        hspaces, radii, total_num_peaks=140
    )

    cands = []
    for a, x, y, r in zip(acc, hcx, hcy, rs):
        x = int(x)
        y = int(y)
        r = int(r)

        if not (0 <= x < w and 0 <= y < h and play_mask[y, x]):
            continue

        # --- centre too close to exact board centre? ------------------------
        dist_c = np.hypot(x - bcx, y - bcy)
        if center_forbid > 0 and dist_c < center_forbid:
            # treat as empty 20-hole / artifact
            continue

        # --- local stats FIRST (needed for Smart Ring on white) -------------
        e_hit = _edge_ring(edges, x, y, r, n=50)
        mu_in, sd_in, mu_out = _inside_stats(lum, x, y, r)
        contrast    = abs(mu_in - mu_out)
        delta_board = abs(mu_in - med)
        mid_std, mid_mean = _angular_uniformity(lum, x, y, r, n=40, frac=0.5)

        # brightness relative to board
        diff_val  = med - mu_in          # >0 darker than board, <0 lighter
        # --- classify brightness --------------------------------------------
        is_dark_disc  = diff_val > 0.05
        is_light_disc = diff_val < -0.02
        is_ghost      = not (is_dark_disc or is_light_disc)

        # --- proximity to any scoring ring ----------------------------------
        on_ring = any(abs(dist_c - rr) < ring_margin for rr in ring_radii)

        # Extra clean-up for the very bright white board:
        # kill board-coloured blobs (ghosts) unless they are insanely "disc-like".
        if preset_name == "bright_white" and is_ghost:
            # Only keep if they are VERY strong edges and clearly different
            # from board/background – real discs should be classified as
            # dark/light, so this almost never triggers for them.
            if not (e_hit >= 0.40 and contrast >= 0.25 and delta_board >= 0.10):
                continue
        relaxed_for_ring = False

        if white_like and on_ring:
            # White-ish boards (bright white + bright mid-wood):
            #   - keep clearly dark/light discs on the line
            #   - kill wood-coloured ghosts or ultra-weak edges
            if is_ghost or e_hit < 0.5 * e_hit_min:
                continue
            relaxed_for_ring = True
        elif on_ring:
            # Non-white-like boards: still hard-suppress candidates on rings
            continue

        # --- brightness-dependent thresholds --------------------------------
        # Make edge + uniformity checks much more forgiving for white-like boards
        base_e_min = e_hit_min
        if white_like:
            base_e_min *= 0.5      # global relaxation for bright_white / mid_wood_soft

        effective_e_min = base_e_min
        if relaxed_for_ring:
            # even more lenient when we *know* it's sitting on a scoring line
            effective_e_min *= 0.7

        eff_sd_in_max   = sd_in_max  * (1.3 if white_like else 1.0)
        eff_mid_std_max = mid_std_max * (1.3 if white_like else 1.0)

        if (
            e_hit       < effective_e_min or
            sd_in       > eff_sd_in_max or
            contrast    < contrast_min or
            delta_board < delta_board_min or
            mid_std     > eff_mid_std_max
        ):
            continue


        cands.append((x, y, r, float(e_hit)))

    if not cands:
        print("[DD] No disc candidates after photometric checks.")
        return []

    # --- Non-max suppression on centres -------------------------------------
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
        print("[DD] All candidates suppressed by NMS.")
        return []

    # --- Colour features & two-cluster k-means ------------------------------
    lab_img = color.rgb2lab(straightened_img.astype(float) / 255.0)
    feats = [
        _lab_patch_mean(
            lab_img, x, y,
            half=min(6, max(3, int(0.5 * r0)))
        )
        for (x, y, r, s) in picked
    ]
    team_labels, centroids = _kmeans2_lab(feats, seed=0)

    r_use = int(round(r0))
    results = []
    for (x, y, r, s), labv, team in zip(picked, feats, team_labels):
        results.append({
            'center': (int(x), int(y)),
            'radius': r_use,
            'score':  float(s),
            'team':   int(team),
            'lab':    labv,
        })

    if cfg.get("debug_show_discs", True):
        debug_show_discs(straightened_img, results, board_result)

    return results[:max_discs]
def _detect_discs_midwood_smart(straightened_img, board_result, config):
    """
    Smart Ring Logic tuned for the mid-wood board (~0.5 brightness).
    Goal: kill ring-only ghosts aggressively, keep real discs (dark or light).
    """
    h, w = straightened_img.shape[:2]
    r0, r_min, r_max = _expected_disc_radius(board_result, config, pad_frac=0.30)

    if r0 is None:
        print("[DD] No center ring radius; cannot infer disc size.")
        return []

    # Board centre & ring radii
    bcx, bcy = board_result['center']
    raw_rings = board_result.get('rings', {})
    ring_radii = []
    for name, rinfo in raw_rings.items():
        if name == 'outer':
            continue
        ring_radii.append(_ring_radius_value(rinfo, default=0.0))

    cfg = config.get('disc_detection', {})
    max_discs = cfg.get('max_discs', 28)

    # --- 1. Board type -------------------------------------------------------
    raw_gray = color.rgb2gray(straightened_img)
    board_median_brightness = float(np.median(raw_gray))
    is_wood_board = board_median_brightness < 0.60

    # --- 2. Thresholds -------------------------------------------------------
    if not is_wood_board:
        # White board path (probably not used for 0.518)
        print(f"[DD] WHITE board ({board_median_brightness:.2f}). Strict mode.")
        THR_EDGE  = 0.25
        THR_DELTA = 0.08
        C_LOW, C_HIGH = 0.05, 0.98
        # The other thresholds are unused in this path
    else:
        print(f"[DD] WOOD board ({board_median_brightness:.2f}). Smart Ring mode.")
        # STRONGER gating to kill ghosts
        THR_EDGE          = 0.19   # min edge strength off-ring
        THR_EDGE_RING     = 0.08   # min edge strength on-ring
        THR_DELTA         = 0.05   # min |disc - board| (brightness delta)
        # Colour-based ghost vs disc split:
        GHOST_COLOR_T     = 0.15   # abs_delta < this ⇒ ghost-coloured
        THR_CONTRAST_DISC = 0.15   # min contrast for real discs
        THR_CONTRAST_GHOST = 0.27  # min contrast for ghost-coloured blobs
        SD_IN_MAX         = 0.20
        MID_STD_MAX       = 0.60
        C_LOW, C_HIGH     = 0.02, 0.40

    # --- 3. Image processing -------------------------------------------------
    img_f = straightened_img.astype(float) / 255.0
    sharp = filters.unsharp_mask(img_f, radius=2, amount=1.5)
    lum   = color.rgb2gray(sharp)
    lum   = exposure.equalize_adapthist(lum, clip_limit=0.01)

    med = float(np.median(lum))

    if is_wood_board:
        low_t, high_t = C_LOW, C_HIGH
    else:
        low_t  = max(0.05, 0.66 * med)
        high_t = min(0.98, 1.33 * med)

    e1 = feature.canny(
        lum, sigma=1.5,
        low_threshold=low_t, high_threshold=high_t
    )
    e2 = feature.canny(
        1.0 - lum, sigma=1.5,
        low_threshold=low_t, high_threshold=high_t
    )
    edges = np.maximum(e1, e2)

    # playable area
    play_mask = _play_area_mask(
        straightened_img.shape,
        board_result,
        inset_px=max(3, int(0.003 * min(h, w)))
    )

    # --- 4. Hough transform --------------------------------------------------
    radii = np.arange(max(2, r_min), max(3, r_max + 1), 1, dtype=int)
    hspaces = transform.hough_circle(edges, radii)
    acc, hcx, hcy, rs = transform.hough_circle_peaks(
        hspaces, radii, total_num_peaks=160
    )

    cands = []
    for a, x, y, r in zip(acc, hcx, hcy, rs):
        x = int(x)
        y = int(y)
        r = int(r)

        if not (0 <= x < w and 0 <= y < h and play_mask[y, x]):
            continue

        # --- local stats -----------------------------------------------------
        e_hit = _edge_ring(edges, x, y, r, n=50)
        mu_in, sd_in, mu_out = _inside_stats(lum, x, y, r)
        mid_std, mid_mean = _angular_uniformity(lum, x, y, r, n=40, frac=0.5)

        diff_val  = med - mu_in        # >0 darker than board, <0 lighter
        abs_delta = abs(diff_val)
        contrast  = abs(mu_in - mu_out)

        # --- which ring? -----------------------------------------------------
        dist_c  = np.hypot(x - bcx, y - bcy)
        on_ring = any(abs(dist_c - rr) < 0.4 * r0 for rr in ring_radii)

        keep_it = False

        if is_wood_board:
            # --- classify by colour similarity to board ----------------------
            is_ghost      = (abs_delta < GHOST_COLOR_T)
            is_dark_disc  = (diff_val >=  GHOST_COLOR_T)
            is_light_disc = (diff_val <= -GHOST_COLOR_T)
            is_disc       = is_dark_disc or is_light_disc

            if on_ring:
                if is_disc:
                    # Real disc on scoring line: easier contrast bar
                    if (
                        e_hit    >= THR_EDGE_RING and
                        abs_delta >= THR_DELTA and
                        contrast >= THR_CONTRAST_DISC
                    ):
                        keep_it = True
                elif is_ghost:
                    # Ghost-coloured on the ring:
                    # only keep if insanely contrasty (almost never)
                    if (
                        e_hit    >= THR_EDGE_RING and
                        abs_delta >= THR_DELTA and
                        contrast >= THR_CONTRAST_GHOST
                    ):
                        keep_it = True
            else:
                # NOT on a ring
                if is_disc:
                    # Real disc off-line: no need for huge contrast
                    if (
                        e_hit    >= THR_EDGE and
                        abs_delta >= THR_DELTA and
                        contrast >= THR_CONTRAST_DISC
                    ):
                        keep_it = True
                elif is_ghost:
                    # Ghost-coloured off-line: require strong contrast
                    if (
                        e_hit    >= THR_EDGE and
                        abs_delta >= THR_DELTA and
                        contrast >= THR_CONTRAST_GHOST
                    ):
                        keep_it = True
        else:
            # White-board fallback (no ghost logic, just strict ring-kill)
            if (not on_ring) and e_hit >= THR_EDGE and abs_delta >= THR_DELTA:
                keep_it = True

        # Global texture gate
        if keep_it:
            if is_wood_board and (sd_in > SD_IN_MAX or mid_std > MID_STD_MAX):
                keep_it = False

        if keep_it:
            cands.append((x, y, r, float(e_hit)))

    if not cands:
        return []

    # --- 5. Non-max suppression ---------------------------------------------
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

    # --- 6. Remove center-hole artifacts ------------------------------------
    cx, cy = board_result['center']
    rings = board_result.get('rings', {})
    r_center_raw = rings.get('center', 0.0)
    r_center = _ring_radius_value(r_center_raw, default=0.0)
    if r_center > 0 and r0 is not None:
        filtered = []
        center_forbid = r_center + 0.5 * r0
        for x, y, r, s in picked:
            dist = np.hypot(x - cx, y - cy)
            if dist <= center_forbid:
                continue
            filtered.append((x, y, r, s))
        picked = filtered

    if not picked:
        return []

    # --- 7. Colour clustering -----------------------------------------------
    lab_img = color.rgb2lab(straightened_img.astype(float) / 255.0)
    feats = [
        _lab_patch_mean(
            lab_img, x, y,
            half=min(6, max(3, int(0.5 * r0)))
        )
        for (x, y, r, s) in picked
    ]
    team_labels, centroids = _kmeans2_lab(feats, seed=0)

    r_use = int(round(r0))
    results = []
    for (x, y, r, s), labv, team in zip(picked, feats, team_labels):
        results.append({
            'center': (int(x), int(y)),
            'radius': r_use,
            'score':  float(s),
            'team':   int(team),
            'lab':    labv,
        })

    if cfg.get("debug_show_discs", True):
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
            rad = _ring_radius_value(r, default=0.0)
            if rad <= 0:
                continue
            circ = patches.Circle(
                (cx, cy), rad,
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


def extract_disc_colours(image, discs, patch_half=3):
    """Extract Lab color features from disc centers."""
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
    """K-means clustering in Lab color space."""
    if features.shape[0] == 0:
        return np.array([], dtype=int), np.zeros((n_clusters, 3))
    rng = np.random.default_rng(seed)
    centroids = [features[rng.integers(0, len(features))]]
    for _ in range(1, n_clusters):
        d2 = np.min([np.sum((features - c) ** 2, axis=1) for c in centroids], axis=0)
        d2_sum = d2.sum()
        if d2_sum < 1e-9:
            centroids.append(features[rng.integers(0, len(features))])
        else:
            probs = d2 / d2_sum
            probs = probs / probs.sum()
            centroids.append(features[rng.choice(len(features), p=probs)])
    centroids = np.stack(centroids, axis=0)

    for _ in range(30):
        d = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(d, axis=1)
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


def remap_teams_by_lightness(labels, centroids):
    """Remap team labels so Team 0 is lighter (higher L) and Team 1 is darker (lower L)."""
    if len(centroids) < 2:
        return labels, centroids
    
    L0 = centroids[0, 0]
    L1 = centroids[1, 0]
    
    if L0 < L1:
        labels = 1 - labels
        centroids = centroids[[1, 0], :]
    
    return labels, centroids


def deltaE_ab(c1, c2):
    """Euclidean distance in ab-plane (ignores L)."""
    return float(np.linalg.norm(c1[1:] - c2[1:]))


def check_colour_similarity(features, labels):
    """Return ΔE between cluster centroids and uncertain indices."""
    if len(features) == 0:
        return 0.0, []

    labs0 = features[labels == 0]
    labs1 = features[labels == 1]
    if len(labs0) == 0 or len(labs1) == 0:
        return 0.0, list(range(len(features)))

    c0 = labs0.mean(axis=0)
    c1 = labs1.mean(axis=0)
    score = deltaE_ab(c0, c1)

    uncertain = []
    for i, f in enumerate(features):
        d0 = np.linalg.norm(f - c0)
        d1 = np.linalg.norm(f - c1)
        if abs(d0 - d1) < 3.0:
            uncertain.append(i)
    return score, uncertain


@dataclass
class DiscScore:
    idx: int
    score: int
    confidence: float
    flags: list  # list of strings


def calculate_disc_scores(discs, mask, line_band, board_info):
    """Calculate score for each disc using perimeter sampling and line rule."""
    results = []
    if len(discs) == 0:
        return results

    h, w = mask.shape
    cx, cy = board_info['center']
    r_center = board_info['radii']['center']
    r_ring5 = board_info['radii']['ring_5']  # Use ring_5 as playable boundary

    for i, d in enumerate(discs):
        x, y = d['center']
        r = d['radius']
        flags = []
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

        if len(vals) == 0:
            results.append(DiscScore(i, 0, 0.0, ['no_samples']))
            continue
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        maj = max(counts.items(), key=lambda kv: kv[1])[0]

        center_dist = np.hypot(x - cx, y - cy)
        fully_in_20 = (center_dist + r) < (r_center - 1.0)

        # Start from majority ring
        base_score = maj
        if maj == 15 and fully_in_20:
            base_score = 20

        # --- LINE TOUCH LOGIC: choose the lowest ring actually touched ------
        if lower_touch:
            # What ring values appear on the disc perimeter?
            # (we only care about the standard scores)
            touched_vals = sorted({v for v in vals if v in (0, 5, 10, 15, 20)})
            if len(touched_vals) >= 2:
                # If we’re really on a boundary (saw 2+ different zones),
                # use the *lowest* of them.
                base_score = touched_vals[0]
            # If len == 1, we’re probably just grazing numerical noise:
            # leave base_score as-is.

        # --- OUTSIDE CHECK: only if the disc is COMPLETELY outside the 5-ring
        outside_margin = 1.0  # pixels of tolerance
        if (center_dist - r) >= (r_ring5 + outside_margin):
            base_score = 0
            flags.append('outside')


        agree = np.mean([v == maj for v in vals])
        conf = float(agree) if 'line_touch' not in flags else max(0.3, float(agree) * 0.8)

        results.append(DiscScore(i, int(base_score), conf, flags))
    return results


def calculate_team_totals(disc_scores, team_labels):
    """Sum visible scores (5/10/15) per team. 20s handled separately."""
    t0 = 0
    t1 = 0
    for ds in disc_scores:
        if ds.score == 20:
            continue
        if team_labels[ds.idx] == 0:
            t0 += ds.score
        else:
            t1 += ds.score
    return t0, t1


def create_results_overlay(image, board_result, discs, labels, disc_scores):
    """Draw rings, discs with team colors, and per-disc scores."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    cx, cy = board_result['center']
    for name, ring_data in board_result['rings'].items():
        r = ring_data['radius'] if isinstance(ring_data, dict) else ring_data
        circ = patches.Circle((cx, cy), r, linewidth=1.5, edgecolor='white', facecolor='none', alpha=0.7)
        ax.add_patch(circ)
        ax.text(cx + r + 4, cy, name, color='black', fontsize=8, weight='bold')

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
    """Calculate round points: 2-0 to winner, 1-1 if tie."""
    if team1_total > team2_total:
        return "2-0"
    elif team2_total > team1_total:
        return "0-2"
    else:
        return "1-1"


def calculate_detection_score(detected_xy, gt_xy, match_threshold=5.0):
    """Calculate disc detection metrics against ground truth."""
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
    """Calculate per-disc score prediction accuracy."""
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
    """Calculate exact match and total error for team scores."""
    t1p, t2p = pred_totals
    t1g, t2g = gt_totals
    exact = int((t1p == t1g) and (t2p == t2g))
    err = abs(t1p - t1g) + abs(t2p - t2g)
    return float(exact), float(err)


def run_complete_pipeline(img_path, config, team1_20s=0, team2_20s=0):
    """Complete pipeline: load image, detect board, detect discs, score, visualize."""
    from skimage import io
    verbose = config.get('verbose', False)
    
    result = {
        'success': False,
        'board_result': None,
        'detected_discs': None,
        'team_assignments': None,
        'team1_score': 0,
        'team2_score': 0,
        'overlay_img': None,
        'error': None
    }
    
    try:
        if verbose:
            print(f"Loading image: {img_path}")
        img_raw = io.imread(img_path)
        img = preprocess_image(img_raw, max_dimension=1200)
        
        if verbose:
            print(f"Working with image dimensions: {img.shape[:2]}")
            print("Detecting edges...")
        edges = detect_edges(img, config)
        
        if verbose:
            print("Detecting board and rings...")
        board_result = detect_board_and_rings(edges, config)
        if board_result is None:
            result['error'] = "Board detection failed"
            if verbose:
                print("❌ Board detection failed")
            return result
        
        result['board_result'] = board_result
        if verbose:
            print(f"✓ Board detected: center={board_result['center']}, radius={board_result['radius']}")
            print(f"  Rings detected: {list(board_result['rings'].keys())}")
        
        # Step 4: Create scoring regions
        if verbose:
            print("Creating scoring regions...")
        scoring_mask = create_scoring_regions(img.shape, board_result)
        raw_boundaries = segmentation.find_boundaries(scoring_mask, mode="inner")

        line_band = morphology.dilation(raw_boundaries, morphology.disk(1))
        
        board_info = {
            "center": board_result["center"],
            "radii": {
                "center": board_result["rings"]["center"]["radius"],
                "ring_5": board_result["rings"]["ring_5"]["radius"],
            },
        }
        
        if verbose:
            print("Detecting discs...")
        detected_discs = detect_discs(img, board_result, config)
        result['detected_discs'] = detected_discs
        
        if len(detected_discs) == 0:
            if verbose:
                print("No discs detected")
            result['success'] = True
            result['overlay_img'] = create_results_overlay(img, board_result, [], [], [])
            return result
        
        if verbose:
            print(f"Detected {len(detected_discs)} discs")
        
        if verbose:
            print("Clustering discs into teams...")
        disc_features = extract_disc_colours(img, detected_discs)
        team_assignments, centroids = cluster_teams_lab(disc_features, config['colour_grouping']['n_clusters'])
        team_assignments, centroids = remap_teams_by_lightness(team_assignments, centroids)
        result['team_assignments'] = team_assignments
        
        if verbose:
            print("Calculating disc scores...")
        disc_scores = calculate_disc_scores(detected_discs, scoring_mask, line_band, board_info)
        
        team1_visible, team2_visible = calculate_team_totals(disc_scores, team_assignments)
        
        team1_final = team1_visible + (team1_20s * 20)
        team2_final = team2_visible + (team2_20s * 20)
        
        result['team1_score'] = team1_final
        result['team2_score'] = team2_final
        
        if verbose:
            print(f"Scores -> Team 1: {team1_final} (visible: {team1_visible}, 20s: {team1_20s})")
            print(f"          Team 2: {team2_final} (visible: {team2_visible}, 20s: {team2_20s})")
        
        if verbose:
            print("Creating visualization overlay...")
        overlay_img = create_results_overlay(img, board_result, detected_discs, team_assignments, disc_scores)
        result['overlay_img'] = overlay_img
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return result

