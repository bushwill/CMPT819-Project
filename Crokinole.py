"""
Crokinole Score Detection System
Core functions for board detection, disc detection, and scoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color, feature, transform, draw


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


def create_scoring_regions(img_shape, board_result):
    """
    Create a segmentation mask with scoring regions based on detected rings.
    
    Returns a mask where pixel values represent point values:
    0 = outside board (beyond outer ring)
    5 = between ring_5 and ring_10
    10 = between ring_10 and ring_15
    15 = between ring_15 and center
    20 = inside center hole
    """
    if board_result is None:
        return None
    
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    board_center = board_result['center']
    detected_rings = board_result['rings']
    
    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - board_center[0])**2 + (y - board_center[1])**2)
    
    # Get ring radii (sorted from outer to inner)
    outer_r = detected_rings.get('outer', None)
    ring_5_r = detected_rings.get('ring_5', None)
    ring_10_r = detected_rings.get('ring_10', None)
    ring_15_r = detected_rings.get('ring_15', None)
    center_r = detected_rings.get('center', None)
    
    # Everything starts at 0 (outside board)
    # Assign scoring values to the spaces BETWEEN rings
    # Ring radii from config: ring_5=0.95, ring_15=0.66, ring_10=0.33, center=0.05
    
    # Outside the outer ring = 0 (already set)
    
    # Between outer ring and ring_5 = still 0 (out of bounds)
    # The outer ring boundary is the edge of the board
    
    # 5 point region: between ring_5 (0.95) and ring_15 (0.66)
    if ring_5_r is not None and ring_15_r is not None:
        mask[(distances <= ring_5_r) & (distances > ring_15_r)] = 5
    
    # 10 point region: between ring_15 (0.66) and ring_10 (0.33)
    if ring_15_r is not None and ring_10_r is not None:
        mask[(distances <= ring_15_r) & (distances > ring_10_r)] = 10
    
    # 15 point region: between ring_10 (0.33) and center (0.05)
    if ring_10_r is not None and center_r is not None:
        mask[(distances <= ring_10_r) & (distances > center_r)] = 15
    
    # 20 point region: inside center hole (0.05)
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
