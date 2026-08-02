"""Built-in preprocessing components grouped by explicit protocol family."""

from salvi.components.preprocessing.column_augmentation import (
    MissingnessIndicators,
    MissingnessIndicatorsConfiguration,
)
from salvi.components.preprocessing.missing_values import (
    MedianModeImputation,
    PreserveMissingValues,
    RejectMissingValues,
)
from salvi.components.preprocessing.numeric_transformation import (
    ZERO_SCALE_TOLERANCE,
    RobustNumericScaling,
)
from salvi.components.preprocessing.source_column_filtering import DropAllMissingColumns

__all__ = [
    "ZERO_SCALE_TOLERANCE",
    "DropAllMissingColumns",
    "MedianModeImputation",
    "MissingnessIndicators",
    "MissingnessIndicatorsConfiguration",
    "PreserveMissingValues",
    "RejectMissingValues",
    "RobustNumericScaling",
]
