"""
Crokinole Score Detection System
Core functions for board detection, disc detection, and scoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color, feature, transform, draw, filters, measure, morphology, exposure
from matplotlib import patches
from dataclasses import dataclass


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
    
    if len(detected_rings) < 4:
        return None
    
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
        'outer': [0, 255, 0],
        'ring_15': [255, 255, 0],
        'ring_10': [255, 165, 0],
        'ring_5': [255, 0, 0],
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


# -----------------------------
# STEP 4: Scoring region mask
# -----------------------------
def create_scoring_regions(image_shape, board_result, ring_dict, line_band_px=4):
    """
    Build scoring mask and a thin buffer band around ring lines for the line-touch rule.

    Returns:
        mask: int32 array with labels {0,5,10,15,20}
        line_band: boolean array marking pixels within ±line_band_px of any boundary
        info: dict with center and radii used
    """
    import numpy as np

    h, w = image_shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    cx, cy = board_result['center']
    cx = float(cx); cy = float(cy)
    dist = np.hypot(xx - cx, yy - cy)

    # Base radii
    r_outer = float(ring_dict.get('outer'))
    # Fallbacks in case some rings weren’t detected:
    r_5    = float(ring_dict.get('ring_5',  0.95 * r_outer))   # boundary between 5 and 10
    r_15   = float(ring_dict.get('ring_15', 0.66 * r_outer))   # boundary between 15 and 10
    r_ctr  = float(ring_dict.get('center',  0.05 * r_outer))   # center hole radius
    # choose outer scoring boundary
    r_play_outer = r_5 if abs(r_outer - r_5) / max(r_outer, 1.0) > 0.08 else min(r_outer, r_5)

    mask = np.zeros((h, w), dtype=np.int32)

    # 20
    mask[dist <= r_ctr] = 20
    # 15
    mask[(dist > r_ctr) & (dist <= r_15)] = 15
    # 10
    mask[(dist > r_15) & (dist <= r_5)] = 10
    # 5 (up to playable outer radius, not board edge/ditch)
    mask[(dist > r_5) & (dist <= r_play_outer)] = 5
    # outside stays 0

    # Line-touch buffer around true scoring boundaries
    boundaries = [r_ctr, r_15, r_5, r_play_outer]

    line_band = np.zeros((h, w), dtype=bool)
    for r in boundaries:
        line_band |= (np.abs(dist - r) <= line_band_px)

    info = {
        'center': (cx, cy),
        'radii': {
            'outer': r_outer,
            'ring_5': r_5,
            'ring_15': r_15,
            'center': r_ctr
        }
    }
    return mask, line_band, info



def visualize_scoring_regions(image, mask):
    """Overlay a transparent color for each scoring region."""
    import matplotlib.pyplot as plt
    cmap = {
        0: (0, 0, 0, 0.0),    # transparent
        5: (1, 0, 0, 0.25),   # red
        10: (1, 0.65, 0, 0.25), # orange
        15: (1, 1, 0, 0.25),  # yellow
        20: (0, 0, 1, 0.25),  # blue
    }
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    for v, col in cmap.items():
        overlay[mask == v] = col

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(overlay)
    ax.set_title("Scoring Regions")
    ax.axis('off')
    plt.show()


# -----------------------------
# STEP 5: Disc detection (Hough)
# -----------------------------
def _play_area_mask(shape, board_result, inset_px=6):
    """
    True only inside the playable wood (<= 5-point ring), excluding the ditch.
    Prefer 'ring_5'. If it's missing, estimate it as 0.95 * outer.
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = board_result['center']
    rings = board_result['rings']

    rout = float(rings.get('outer', 0.0))
    # KEY FIX: if ring_5 not present, estimate as 0.95*outer (not 'outer' itself)
    r5 = float(rings.get('ring_5', 0.95 * rout if rout > 0 else 0.0))

    # If both exist, choose the smaller as the playable limit (safest)
    r_play_outer = r5 if r5 > 0 else rout
    if rout > 0 and r5 > 0:
        r_play_outer = min(r5, rout)

    r_play_outer = max(0.0, r_play_outer - inset_px)
    dist = np.hypot(xx - cx, yy - cy)
    return dist <= r_play_outer

def _perimeter_edge_score(edge_img, x, y, r, n=48):
    """Fraction of perimeter samples that land on an edge."""
    h, w = edge_img.shape
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.clip((x + r*np.cos(ang)).round().astype(int), 0, w-1)
    ys = np.clip((y + r*np.sin(ang)).round().astype(int), 0, h-1)
    return float(edge_img[ys, xs].mean())

def _interior_stats(lum, x, y, r):
    """Mean/std inside disc and mean in a thin annulus just outside."""
    h, w = lum.shape
    yy, xx = np.ogrid[:h, :w]
    d = np.hypot(xx - x, yy - y)
    inside = d <= (r - 1)
    ann    = (d > (r + 1)) & (d <= (r + 4))  # thin ring just outside disc
    if not inside.any() or not ann.any():
        return 0.0, 1.0, 0.0
    mu_in  = float(lum[inside].mean())
    sd_in  = float(lum[inside].std())
    mu_out = float(lum[ann].mean())
    return mu_in, sd_in, mu_out

def _wood_mask(image, min_L=0.15, min_chroma=12.0):
    """
    Return boolean mask of 'wood-like' pixels:
    - Not very dark (L > min_L, L in [0,1])
    - With some chroma (sqrt(a^2 + b^2) > min_chroma)
    Tuned to keep the brown board and exclude the black ditch.
    """
    img = image.astype(float) / 255.0
    lab = color.rgb2lab(img)  # L ~ [0,100]
    L = lab[..., 0] / 100.0
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    mask = (L > min_L) & (chroma > min_chroma)
    # Clean up: remove small specks and fill small holes
    mask = morphology.remove_small_objects(mask, 500)
    mask = morphology.remove_small_holes(mask, 500)
    return mask


def _playable_mask_robust(image, board_result, inset_px=2):
    circle = _play_area_mask(image.shape, board_result, inset_px=inset_px)
    # wood mask is optional – use only if it keeps at least 20% of the circle
    try:
        wood = _wood_mask(image)
        combo = circle & wood
        frac_circle = float(np.mean(circle))
        frac_combo  = float(np.mean(combo))
        # if wood mask keeps too little of the circle, ignore it
        if frac_circle > 0 and (frac_combo / frac_circle) >= 0.2:
            return combo
    except Exception:
        pass
    return circle


def _perimeter_edge_score(edge_img, x, y, r, n=48):
    h, w = edge_img.shape
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.clip((x + r*np.cos(ang)).round().astype(int), 0, w-1)
    ys = np.clip((y + r*np.sin(ang)).round().astype(int), 0, h-1)
    return float(edge_img[ys, xs].mean())

def _interior_stats(lum, x, y, r):
    h, w = lum.shape
    yy, xx = np.ogrid[:h, :w]
    d = np.hypot(xx - x, yy - y)
    inside = d <= (r - 1)
    ann    = (d > (r + 1)) & (d <= (r + 4))
    if not inside.any() or not ann.any():
        return 0.0, 1.0, 0.0
    mu_in  = float(lum[inside].mean())
    sd_in  = float(lum[inside].std())
    mu_out = float(lum[ann].mean())
    return mu_in, sd_in, mu_out

def _edge_contiguity(edge_img, x, y, r, n=72):
    """Longest contiguous run of edge hits along the circle, / n."""
    h, w = edge_img.shape
    ang = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = np.clip((x + r*np.cos(ang)).round().astype(int), 0, w-1)
    ys = np.clip((y + r*np.sin(ang)).round().astype(int), 0, h-1)
    hits = edge_img[ys, xs].astype(np.uint8)

    # handle wrap-around by doubling then taking max run
    arr = np.concatenate([hits, hits])
    best = cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    best = min(best, n)  # cap to one revolution
    return best / float(n)

def _ringline_pressure(edge_img, x, y, r, inner=0, outer=5):
    """
    Edge density in a thin outer annulus [r+inner, r+outer].
    Ring lines produce high density; true discs have a clean ring.
    Returns fraction in [0,1].
    """
    h, w = edge_img.shape
    yy, xx = np.ogrid[:h, :w]
    d = np.hypot(xx - x, yy - y)
    ann = (d >= (r + inner)) & (d <= (r + outer))
    if not np.any(ann):
        return 0.0
    return float(edge_img[ann].mean())



def detect_discs(straightened_img, board_result, config):
    """
    Robust disc detection:
      - unsharp + CLAHE to boost edges
      - adaptive Canny thresholds
      - Hough circles on both normal and inverted edges
      - LoG (blob_log) as a second detector
      - NMS, playable-area mask, and limit to max_discs
    Returns: list of {'center': (x,y), 'radius': r, 'score': a}
    """
    cfg = config['disc_detection']
    h, w = straightened_img.shape[:2]
    min_dim = min(h, w)

    # radius range (tighten a bit around the expected disc size)
    r_min = int(max(2, cfg['min_radius_ratio'] * min_dim))
    r_max = int(max(r_min + 2, cfg['max_radius_ratio'] * min_dim))

    # Sharpen and equalize
    img = straightened_img.astype(float) / 255.0
    sharp = filters.unsharp_mask(img, radius=2, amount=1.5)
    lum = color.rgb2gray(sharp)
    lum = exposure.equalize_adapthist(lum, clip_limit=0.01)

    # Adaptive Canny thresholds
    med = np.median(lum)
    low_t = max(0.05, med * 0.66)
    high_t = min(0.98, med * 1.33)

    edges = feature.canny(lum, sigma=1.5, low_threshold=low_t, high_threshold=high_t)
    edges_inv = feature.canny(1.0 - lum, sigma=1.5, low_threshold=low_t, high_threshold=high_t)

    # Hough on both polarities
    def _hough_pass(edge_img, total_peaks, thr):
        radii = np.arange(r_min, r_max + 1, 1)
        hspaces = transform.hough_circle(edge_img, radii)
        acc, cx, cy, rs = transform.hough_circle_peaks(hspaces, radii, total_num_peaks=total_peaks)
        return [(int(x), int(y), int(r), float(a)) for a, x, y, r in zip(acc, cx, cy, rs) if float(a) >= thr]

    strict = _hough_pass(edges, total_peaks=80, thr=cfg['strict_threshold'])
    strict += _hough_pass(edges_inv, total_peaks=80, thr=cfg['strict_threshold'])

    cands = strict
    if len(cands) < 8:  # too few → fallback
        loose = _hough_pass(edges, total_peaks=140, thr=cfg['fallback_threshold'])
        loose += _hough_pass(edges_inv, total_peaks=140, thr=cfg['fallback_threshold'])
        cands += loose

    # LoG blobs (second detector), convert sigmas to radii ~ sqrt(2)*sigma
    blobs = feature.blob_log(lum, min_sigma=r_min/np.sqrt(2), max_sigma=r_max/np.sqrt(2),
                             num_sigma=10, threshold=0.02)
    for y, x, sigma in blobs:
        r = int(round(np.sqrt(2) * float(sigma)))
        cands.append((int(x), int(y), r, 0.5))  # give a modest score

    # Non-max suppression by center spacing (conservative to kill duplicates)
    cands = sorted(cands, key=lambda d: d[3], reverse=True)
    print("[DD] initial candidates:", len(cands))
    keep = []
    for x, y, r, s in cands:
        ok = True
        for x2, y2, r2, s2 in keep:
            # slightly tighter than before to reduce dupes
            if np.hypot(x - x2, y - y2) < 1.0 * max(r, r2):
                ok = False
                break
        if ok:
            keep.append((x, y, r, s))
    if not keep:
        return []
    print("[DD] after NMS:", len(keep))


    # ---------- Adaptive radius band around dominant size ----------
    radii = np.array([r for _, _, r, _ in keep], dtype=float)
    r_med = float(np.median(radii))
    iqr   = float(np.percentile(radii, 75) - np.percentile(radii, 25))
    if iqr <= 0:
        iqr = max(1.0, 0.1 * r_med)

    # primary band (±0.9*IQR) but also clamp to ±25% of the median
    r_lo_h = r_med - 0.9 * iqr
    r_hi_h = r_med + 0.9 * iqr
    r_lo_p = 0.80 * r_med
    r_hi_p = 1.25 * r_med
    r_lo = max(r_min, min(r_lo_h, r_lo_p))
    r_hi = min(r_max, max(r_hi_h, r_hi_p))

    keep = [(x, y, r, s) for (x, y, r, s) in keep if (r_lo <= r <= r_hi)]
    if not keep:
        # last resort: just use the ±25% band
        r_lo, r_hi = max(r_min, r_lo_p), min(r_max, r_hi_p)
        keep = [(x, y, r, s) for (x, y, r, s) in cands if (r_lo <= r <= r_hi)]
    if not keep:
        return []



    # ---------- Build robust playable mask ----------
    edges_strong = np.maximum(edges, edges_inv)
    mask_play = _playable_mask_robust(straightened_img, board_result,
                                      inset_px=max(3, int(0.003 * min_dim)))
    print("[DD] playable mask true frac:",
      float(np.mean(_playable_mask_robust(straightened_img, board_result,
                                          inset_px=max(3, int(0.003*min_dim))))))


    # ---------- Compute features for adaptive thresholds ----------
    feats = []
    for x, y, r, s in keep:
        if not (0 <= x < w and 0 <= y < h and mask_play[y, x]):
            continue
        e_score = _perimeter_edge_score(edges_strong, x, y, r, n=50)
        mu_in, sd_in, mu_out = _interior_stats(lum, x, y, r)
        contrast = abs(mu_in - mu_out)
        feats.append((x, y, r, s, e_score, sd_in, contrast))

    if not feats:
        return []
    print("[DD] feats inside playable area:", len(feats))


    # ---------- Adaptive thresholds (looser floors based on your stats) ----------
    e_scores  = np.array([f[4] for f in feats])  # perimeter edge hit rate
    sd_vals   = np.array([f[5] for f in feats])  # interior std
    ctr_vals  = np.array([f[6] for f in feats])  # interior vs ring contrast

    # Floors/Ceilings tuned for weak-edges images
    e_thr_floor = 0.04   # was 0.18 — your p50 is ~0.04
    sd_thr_ceil = 0.18   # slightly stricter texture than 0.20
    c_thr_floor = 0.06   # need some contrast, but not huge

    e_thr  = max(e_thr_floor, float(np.percentile(e_scores, 20)))
    sd_thr = min(sd_thr_ceil, float(np.percentile(sd_vals, 85)))
    c_thr  = max(c_thr_floor, float(np.percentile(ctr_vals, 25)))

    prelim = [(x, y, r, s, es, sd, ct) for (x, y, r, s, es, sd, ct) in feats
          if (es >= e_thr and sd <= sd_thr and ct >= c_thr)]
    if not prelim:
        # last-resort fallback: circle-only mask + loose thresholds
        mask_play_soft = _play_area_mask(straightened_img.shape, board_result,
                                        inset_px=max(2, int(0.002*min_dim)))
        feats2 = []
        for x, y, r, s in keep:
            if not (0 <= x < w and 0 <= y < h and mask_play_soft[y, x]):
                continue
            es = _perimeter_edge_score(edges_strong, x, y, r, n=36)
            mu_in, sd_in, mu_out = _interior_stats(lum, x, y, r)
            ct = abs(mu_in - mu_out)
            feats2.append((x, y, r, s, es, sd_in, ct))
        if feats2:
            e_scores2 = np.array([f[4] for f in feats2])
            sd_vals2  = np.array([f[5] for f in feats2])
            ctr_vals2 = np.array([f[6] for f in feats2])
            e_thr  = max(0.12, float(np.percentile(e_scores2, 20)))
            sd_thr = min(0.30, float(np.percentile(sd_vals2, 80)))
            c_thr  = max(0.010, float(np.percentile(ctr_vals2, 30)))
            prelim = [(x,y,r,s,es,sd,ct) for (x,y,r,s,es,sd,ct) in feats2
                    if (es >= e_thr and sd <= sd_thr and ct >= c_thr)]

    if not prelim:
        return []
    print("[DD] prelim survivors:", len(prelim),
      "| e_thr:", round(e_thr,3), "sd_thr:", round(sd_thr,3), "c_thr:", round(c_thr,3))

    # ---------- Merge nearby survivors again (pick best edge score per group) ----------
    prelim.sort(key=lambda t: t[4], reverse=True)  # sort by e_score
    merged = []
    for cand in prelim:
        x, y, r, s, es, sd, ct = cand
        keep_it = True
        for (X, Y, R, ES) in merged:
            if np.hypot(x - X, y - Y) < 0.9 * max(r, R):
                keep_it = False
                break
        if keep_it:
            merged.append((x, y, r, es))
    

    print("[DD] merged:", len(merged))



    # ---------- Build output ----------
    filtered = [{'center': (int(x), int(y)), 'radius': int(r), 'score': float(es)}
                for (x, y, r, es) in merged]

    # Cap to max_discs
    filtered = filtered[:cfg['max_discs']]
    # In detect_discs temporarily print:
    print("e_scores p25/p50:", np.percentile(e_scores, 25), np.percentile(e_scores, 50))
    print("sd_vals  p50/p75:", np.percentile(sd_vals, 50),  np.percentile(sd_vals, 75))
    print("ctr_vals p35/p50:", np.percentile(ctr_vals, 35), np.percentile(ctr_vals, 50))

    return filtered



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


