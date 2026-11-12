"""Stage 4 quantitative analysis (Qt-free).

Educational metrics only — not calibrated clinical PET quantities.
"""

from petsim.analysis.coordinates import (
    fov_to_pixel,
    image_axes,
    pixel_to_fov,
)
from petsim.analysis.count_comparison import (
    DEFAULT_COUNT_LEVELS,
    CountComparisonResult,
    CountLevelMetrics,
    compare_by_coincidence_count,
)
from petsim.analysis.localization import (
    LocalizationResult,
    localization_error,
    localize_tumor,
)
from petsim.analysis.pipeline import (
    QuantitativeAnalysisResult,
    analyze_reconstruction,
)
from petsim.analysis.roi import (
    ContrastResult,
    ROIMasks,
    build_tumor_background_rois,
    compute_contrast_metrics,
)

__all__ = [
    "ContrastResult",
    "CountComparisonResult",
    "CountLevelMetrics",
    "DEFAULT_COUNT_LEVELS",
    "LocalizationResult",
    "QuantitativeAnalysisResult",
    "ROIMasks",
    "analyze_reconstruction",
    "build_tumor_background_rois",
    "compare_by_coincidence_count",
    "compute_contrast_metrics",
    "fov_to_pixel",
    "image_axes",
    "localization_error",
    "localize_tumor",
    "pixel_to_fov",
]
