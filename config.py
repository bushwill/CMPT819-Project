"""
Crokinole Detection Configuration
Central configuration file for all detection parameters.
"""

# Default configuration for all scripts
DEFAULT_CONFIG = {
    'verbose': False,
    'contrast_boost': 1.2,
    'board_detection': {
        'min_circle_ratio': 0.25,
        'max_circle_ratio': 0.9,
        'radius_step': 5,
        'canny_low_threshold': 0.1,
        'canny_high_threshold': 0.2,
        'edge_sigma': 2,
        'hough_threshold': 0.5
    },
    'ring_ratios': {
        'ring_5': 1.00,
        'ring_10': 0.66,
        'ring_15': 0.33,
        'center': 0.05
    },
    'ring_search': {
        'tolerance': 0.08,
        'step_size': 2,  # Minimum step size (adaptive scaling maintains ~200 radii for performance)
        'max_center_offset': 0.15,
        'top_peaks': 5
    },
    'board_validation': {
        'min_rings_required': 4
    },
    'disc_detection': {
        'min_radius_ratio': 0.02,
        'max_radius_ratio': 0.06,
        'max_discs': 28,
        'strict_threshold': 0.6,
        'fallback_threshold': 0.4,
        'min_disc_spacing': 0.6
    },
    'colour_grouping': {
        'n_clusters': 2,
        'similarity_threshold': 0.3,
    },
}
