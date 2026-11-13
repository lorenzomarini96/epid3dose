
"""
Global constants used throughout the epid3dose package.

This module centralizes:
- Model input dimensions
- CT preprocessing parameters
- EPID preprocessing and normalization parameters
- RTDOSE normalization parameters
- Model architecture configuration
"""

# =============================================================================
# MODEL INPUT DIMENSIONS
# =============================================================================

CT_SHAPE = (128, 128, 128)
EPID_SHAPE = (256, 256)
DOSE_SHAPE = (128, 128, 128)


# =============================================================================
# CT PREPROCESSING
# =============================================================================

CT_HU_MIN = -1000.0
CT_HU_MAX = 1000.0

CT_NORMALIZATION_MIN = 0.0
CT_NORMALIZATION_MAX = 1.0


# =============================================================================
# EPID PREPROCESSING
# =============================================================================

# Native detector pixel spacing
EPID_DETECTOR_PIXEL_SPACING = 0.405  # mm/pixel

# EPID pixel spacing after crop and resize to 256 x 256
EPID_FINAL_PIXEL_SPACING = 1.347  # mm/pixel

# Original Portal Dose geometry
PD_PIXEL_SPACING = 1.0  # mm/pixel
EPID_CROP_PD_SHAPE = (345, 345)

# Private DICOM tag containing the detector PSF
EPID_PSF_TAG = (0x0021, 0x1002)

# Global normalization values calculated during training
EPID_MIN = 1283.7572954296636
EPID_MAX = 1584183.7185058743


# =============================================================================
# RTDOSE PREPROCESSING
# =============================================================================

# Global normalization values calculated on the training dataset
DOSE_GLOBAL_MIN = 0       # To be recovered from the original training pipeline
DOSE_GLOBAL_MAX = 6.916368  # Gy


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

INITIAL_FILTERS = 32
OUTPUT_ACTIVATION = "softplus"
