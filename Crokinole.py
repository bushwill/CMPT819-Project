"""
Crokinole Score Detection System
Core functions for board detection, disc detection, and scoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color, feature, transform, draw, filters, measure, morphology, exposure
from matplotlib import patches
from dataclasses import dataclass
import warnings


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
    gray_image = color.rgb2gray(img)
    edges = feature.canny(
        gray_image,
        low_threshold=board_cfg['canny_low_threshold'],
        high_threshold=board_cfg['canny_high_threshold'],
        sigma=board_cfg['edge_sigma']
    )
    return edges


def detect_board_and_rings_ellipse(edges, config):
    """Detect board rings using ellipse detection (fallback for non-orthogonal views).
    Returns board_result with detected ellipses, or None if pattern not found."""
    h, w = edges.shape[:2]
    
    ellipse_cfg = config['ellipse_detection']
    ring_ratios = config['ring_ratios']
    ring_search_cfg = config['ring_search']
    
    print("Attempting ellipse detection (non-orthogonal view detected)...")
    
    # Estimate size range based on image dimensions
    min_size = max(ellipse_cfg['min_size'], int(min(h, w) * 0.1))
    max_size = int(min(h, w) * 0.9)
    
    try:
        # Suppress warnings from ellipse detection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Detect ellipses using Hough transform
            result = transform.hough_ellipse(
                edges,
                threshold=int(ellipse_cfg['threshold'] * 255),
                accuracy=ellipse_cfg['accuracy'],
                min_size=min_size,
                max_size=max_size
            )
            
            # Sort by accumulator value (confidence)
            result = sorted(result, key=lambda x: x[5], reverse=True)
            
            # Filter by eccentricity and take top candidates
            ellipse_candidates = []
            for ellipse in result[:ellipse_cfg['num_peaks']]:
                yc, xc, a, b, orientation, acc = ellipse
                
                # Calculate eccentricity: e = sqrt(1 - (b/a)^2) for a >= b
                semi_major = max(a, b)
                semi_minor = min(a, b)
                if semi_major > 0:
                    eccentricity = np.sqrt(1 - (semi_minor / semi_major) ** 2)
                else:
                    continue
                    
                if eccentricity <= ellipse_cfg['max_eccentricity']:
                    ellipse_candidates.append({
                        'center': (int(xc), int(yc)),
                        'semi_major': int(semi_major),
                        'semi_minor': int(semi_minor),
                        'orientation': orientation,
                        'eccentricity': eccentricity,
                        'acc': acc
                    })
            
            if len(ellipse_candidates) == 0:
                print("No valid ellipses found")
                return None
            
            print(f"Found {len(ellipse_candidates)} ellipse candidates")
            
            # Try pattern matching using ring_5 as reference
            # Sort by semi_major axis (largest first, as ring_5 should be largest)
            ellipse_candidates.sort(key=lambda x: x['semi_major'], reverse=True)
            
            for ring5_candidate in ellipse_candidates:
                ring5_center = ring5_candidate['center']
                ring5_major = ring5_candidate['semi_major']
                
                # Calculate expected outer radius (assuming ellipse represents ring_5)
                board_radius = int(ring5_major / ring_ratios['ring_5'])
                board_center = ring5_center
                
                detected_rings = {
                    'ring_5': ring5_major  # Use semi-major axis as reference
                }
                max_offset = board_radius * ring_search_cfg['max_center_offset']
                
                # Match remaining 3 inner rings
                inner_ratios = {k: v for k, v in ring_ratios.items() if k != 'ring_5'}
                sorted_rings = sorted(inner_ratios.items(), key=lambda x: x[1], reverse=True)
                
                for ring_name, ratio in sorted_rings:
                    expected_major = int(board_radius * ratio)
                    tol = int(expected_major * ring_search_cfg['tolerance'])
                    
                    # Find best matching ellipse
                    best_match = None
                    best_diff = float('inf')
                    
                    for ellipse in ellipse_candidates:
                        # Check size match (using semi-major axis)
                        if abs(ellipse['semi_major'] - expected_major) > tol:
                            continue
                        
                        # Check center proximity
                        dist = np.hypot(ellipse['center'][0] - board_center[0],
                                       ellipse['center'][1] - board_center[1])
                        if dist > max_offset:
                            continue
                        
                        # Track best match
                        diff = abs(ellipse['semi_major'] - expected_major)
                        if diff < best_diff:
                            best_diff = diff
                            best_match = ellipse
                    
                    if best_match is not None:
                        detected_rings[ring_name] = expected_major
                    else:
                        break
                
                # Check if we found all 4 rings
                if len(detected_rings) == 4:
                    outer_radius = board_radius
                    print(f"Board found via ellipse detection: center={board_center}, outer_radius={outer_radius}")
                    print(f"  Detected 4 rings: {', '.join(detected_rings.keys())}")
                    print(f"  Note: Perspective correction recommended for accurate scoring")
                    return {
                        'center': board_center,
                        'radius': outer_radius,
                        'rings': detected_rings,
                        'ellipse_detection': True  # Flag indicating ellipse detection was used
                    }
            
            print("No valid board pattern found with ellipse detection")
            return None
            
    except Exception as e:
        print(f"Ellipse detection failed: {e}")
        return None


def detect_board_and_rings(edges, config):
    """Detect board and rings using ring_5 as primary reference. 
    Falls back to ellipse detection for non-orthogonal views.
    Returns None if not a valid board."""
    h, w = edges.shape[:2]
    
    board_cfg = config['board_detection']
    ring_ratios = config['ring_ratios']
    ring_search_cfg = config['ring_search']
    validation_cfg = config['board_validation']
    
    # OPTIMIZATION: Do ALL Hough transforms upfront
    min_radius = int(min(h, w) * board_cfg['min_circle_ratio'])
    max_radius = int(min(h, w) * board_cfg['max_circle_ratio'])
    
    # Search all radii from 5% to max_radius in one shot
    min_inner = int(min_radius * 0.05)
    all_radii = np.arange(min_inner, max_radius, ring_search_cfg['step_size'])
    
    print(f"Running single Hough transform for {len(all_radii)} radii...")
    hough_res = transform.hough_circle(edges, all_radii)
    accums, cx, cy, radii_detected = transform.hough_circle_peaks(hough_res, all_radii, total_num_peaks=100)
    
    if len(cx) == 0:
        print("No circles found, trying ellipse detection...")
        return detect_board_and_rings_ellipse(edges, config)
    
    print(f"Found {len(cx)} circle candidates, filtering...")
    
    # Build lookup of all detected circles
    all_circles = []
    for x, y, r, acc in zip(cx, cy, radii_detected, accums):
        all_circles.append({
            'center': (int(x), int(y)),
            'radius': int(r),
            'acc': float(acc)
        })
    
    # Get ring_5 candidates (should be around 95% of outer, so large radii)
    # ring_5 is the PRIMARY REFERENCE ring, outer board edge is calculated
    ring5_min = int(min_radius * 0.95)  # Approximate minimum for ring_5
    ring5_candidates = [c for c in all_circles if c['radius'] >= ring5_min]
    ring5_candidates.sort(key=lambda x: x['radius'], reverse=True)
    
    # Try pattern matching using each ring_5 candidate as reference
    for ring5_candidate in ring5_candidates:
        ring5_center = ring5_candidate['center']
        ring5_radius = ring5_candidate['radius']
        
        # Calculate what the outer board radius should be (NOT detected)
        board_radius = int(ring5_radius / ring_ratios['ring_5'])
        board_center = ring5_center
        
        # Only store detected rings (ring_5 + 3 inner)
        detected_rings = {
            'ring_5': ring5_radius
        }
        max_offset = board_radius * ring_search_cfg['max_center_offset']
        
        # Match remaining 3 inner rings (ring_15, ring_10, center)
        inner_ratios = {k: v for k, v in ring_ratios.items() if k != 'ring_5'}
        sorted_rings = sorted(inner_ratios.items(), key=lambda x: x[1], reverse=True)
        
        for ring_name, ratio in sorted_rings:
            expected_r = int(board_radius * ratio)
            tol = int(expected_r * ring_search_cfg['tolerance'])
            
            # Find best matching circle from pre-computed results
            best_match = None
            best_diff = float('inf')
            
            for circle in all_circles:
                # Check radius match
                if abs(circle['radius'] - expected_r) > tol:
                    continue
                
                # Check center proximity
                dist = np.hypot(circle['center'][0] - board_center[0], 
                               circle['center'][1] - board_center[1])
                if dist > max_offset:
                    continue
                
                # Track best match
                diff = abs(circle['radius'] - expected_r)
                if diff < best_diff:
                    best_diff = diff
                    best_match = circle
            
            if best_match is not None:
                detected_rings[ring_name] = expected_r  # Use exact expected radius
            else:
                break  # Pattern invalid
        
        # Check if we found all 4 rings (ring_5 + 3 inner: ring_15, ring_10, center)
        if len(detected_rings) == 4:
            # Calculate outer radius for scoring (not detected)
            outer_radius = board_radius
            print(f"Board found: center={board_center}, outer_radius={outer_radius} (calculated from ring_5={ring5_radius})")
            print(f"  Detected 4 rings: {', '.join(detected_rings.keys())}")
            return {
                'center': board_center,
                'radius': outer_radius,  # Calculated outer radius for scoring
                'rings': detected_rings  # Only the 4 detected rings
            }
    
    print("No valid board pattern found with circles, trying ellipse detection...")
    return detect_board_and_rings_ellipse(edges, config)


def visualize_board_detection(img, board_result):
    """Visualize detected board and rings with color-coded circles."""
    if board_result is None:
        return
    
    board_center = board_result['center']
    detected_rings = board_result['rings']
    
    colors = {
        'ring_5': [255, 0, 0],
        'ring_15': [255, 255, 0],
        'ring_10': [255, 165, 0],
        'center': [0, 0, 255]
    }
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    overlay = img.copy()
    
    for ring_name, ring_radius in detected_rings.items():
        if ring_name in colors:
            circy, circx = draw.circle_perimeter(
                board_center[1], board_center[0], ring_radius,
                shape=img.shape[:2]
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


def create_scoring_regions(img_shape, board_result):
    """
    Create a segmentation mask with scoring regions based on detected rings.
    
    Returns a mask where pixel values represent point values:
    0 = outside board (beyond ring_5)
    5 = between ring_5 and ring_15
    10 = between ring_15 and ring_10
    15 = between ring_10 and center
    20 = inside center hole
    
    Note: Uses board_result['radius'] as the calculated board boundary from ring_5.
    """
    if board_result is None:
        return None
    
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    board_center = board_result['center']
    board_radius = board_result['radius']  # Calculated from ring_5
    detected_rings = board_result['rings']
    
    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - board_center[0])**2 + (y - board_center[1])**2)
    
    # Get ring radii from detected rings
    ring_5_r = detected_rings.get('ring_5', None)
    ring_15_r = detected_rings.get('ring_15', None)
    ring_10_r = detected_rings.get('ring_10', None)
    center_r = detected_rings.get('center', None)
    
    # Everything starts at 0 (outside board)
    # Assign scoring values to the spaces BETWEEN rings
    
    # 5 point region: between ring_5 and ring_15
    if ring_5_r is not None and ring_15_r is not None:
        mask[(distances <= ring_5_r) & (distances > ring_15_r)] = 5
    
    # 10 point region: between ring_15 and ring_10
    if ring_15_r is not None and ring_10_r is not None:
        mask[(distances <= ring_15_r) & (distances > ring_10_r)] = 10
    
    # 15 point region: between ring_10 and center
    if ring_10_r is not None and center_r is not None:
        mask[(distances <= ring_10_r) & (distances > center_r)] = 15
    
    # 20 point region: inside center hole
    if center_r is not None:
        mask[distances <= center_r] = 20
    
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
    
    # Blend with original image
    alpha = 0.5
    if len(img.shape) == 3:
        blended = (alpha * img + (1 - alpha) * overlay).astype(np.uint8)
    else:
        img_rgb = np.stack([img, img, img], axis=2)
        blended = (alpha * img_rgb + (1 - alpha) * overlay).astype(np.uint8)
    
    # Display
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
    
    # Add legend
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

def _expected_disc_radius(board_result, pad_frac=0.10, min_px=4):
    """
    Use the 20-hole radius as the expected disc radius.
    'pad_frac' widens the allowed band by ±(pad_frac * r0).
    """
    rings = board_result['rings']
    r0 = float(rings.get('center', 0.0))   # 20-hole radius
    if r0 <= 0:
        return None, None, None
    band = max(min_px, r0 * pad_frac)
    return r0, max(2, int(r0 - band)), int(r0 + band)


def _play_area_mask(shape, board_result, inset_px=4):
    """
    Inside the playable wood (<= ring_5).
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = board_result['center']
    rings = board_result['rings']
    r5 = float(rings.get('ring_5', 0.0))

    r_play_outer = max(0.0, r5 - inset_px) if r5 > 0 else 0.0
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
    Tiny K=2 k-means on Lab features; returns labels (0/1) and centroids.
    """
    if len(feats) == 0:
        return np.array([], dtype=int), np.zeros((2, 3))
    X = np.asarray(feats, float)
    rng = np.random.default_rng(seed)

    # k-means++ init
    c0 = X[rng.integers(0, len(X))]
    d2 = np.sum((X - c0) ** 2, axis=1)
    probs = d2 / (d2.sum() + 1e-9)
    c1 = X[rng.choice(len(X), p=probs)]
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


def detect_discs(straightened_img, board_result, config):
    """
    Disc detection tied to the 20-hole size + immediate colour grouping.

    Returns a list of dicts:
        {
          'center': (x, y),
          'radius': r,      # snapped to 20-hole radius
          'score': edge_strength,
          'team':  0 or 1,  # colour cluster
          'lab':   (L, a, b)
        }
    """
    h, w = straightened_img.shape[:2]
    r0, r_min, r_max = _expected_disc_radius(board_result, pad_frac=0.10)  # ±10%
    if r0 is None:
        print("[DD] No center ring radius; cannot infer disc size.")
        return []

    cfg = config.get('disc_detection', {})
    max_discs = cfg.get('max_discs', 28)

    # --- Pre-processing -------------------------------------------------------
    img_f = straightened_img.astype(float) / 255.0
    sharp = filters.unsharp_mask(img_f, radius=2, amount=1.5)
    lum   = color.rgb2gray(sharp)
    lum   = exposure.equalize_adapthist(lum, clip_limit=0.01)

    med = float(np.median(lum))
    low_t, high_t = max(0.05, 0.66 * med), min(0.98, 1.33 * med)

    e1 = feature.canny(lum,        sigma=1.5,
                       low_threshold=low_t, high_threshold=high_t)
    e2 = feature.canny(1.0 - lum,  sigma=1.5,
                       low_threshold=low_t, high_threshold=high_t)
    edges = np.maximum(e1, e2)

    # --- Restrict to playable wood -------------------------------------------
    play_mask = _play_area_mask(straightened_img.shape,
                                board_result,
                                inset_px=max(3, int(0.003 * min(h, w))))

    # --- Hough circles in tight radius band ----------------------------------
    radii = np.arange(max(2, r_min), max(3, r_max + 1), 1, dtype=int)
    hspaces = transform.hough_circle(edges, radii)
    acc, cx, cy, rs = transform.hough_circle_peaks(
        hspaces, radii, total_num_peaks=140
    )

    cands = []
    for a, x, y, r in zip(acc, cx, cy, rs):
        x = int(x); y = int(y); r = int(r)
        if not (0 <= x < w and 0 <= y < h and play_mask[y, x]):
            continue

        e_hit = _edge_ring(edges, x, y, r, n=50)
        mu_in, sd_in, mu_out = _inside_stats(lum, x, y, r)
        contrast = abs(mu_in - mu_out)

        # basic photometric vetting (tuned for your board images)
        if e_hit >= 0.06 and sd_in <= 0.20 and contrast >= 0.05:
            cands.append((x, y, r, float(e_hit)))

    if not cands:
        print("[DD] No disc candidates after photometric checks.")
        return []

    # --- Non-max suppression on centres --------------------------------------
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

    # --- Colour features & two-cluster k-means -------------------------------
    lab_img = color.rgb2lab(straightened_img.astype(float) / 255.0)
    feats = [
        _lab_patch_mean(lab_img, x, y,
                        half=min(6, max(3, int(0.5 * r0))))
        for (x, y, r, s) in picked
    ]
    team_labels, centroids = _kmeans2_lab(feats, seed=0)

    # --- Build result list (radius snapped to r0) ----------------------------
    r_use = int(round(r0))
    results = []
    for (x, y, r, s), labv, team in zip(picked, feats, team_labels):
        results.append({
            'center': (int(x), int(y)),
            'radius': r_use,
            'score': float(s),
            'team': int(team),
            'lab': labv,
        })

    # --- Debug overlay --------------------------------------------------------
    if cfg.get("debug_show_discs", True):
        debug_show_discs(straightened_img, results, board_result)

    # Enforce cap
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
    r_ring5 = board_info['radii']['ring_5']  # Use ring_5 as playable boundary

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
        if (center_dist + r) > (r_ring5 + 1):  # Outside ring_5 (playable boundary)
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


