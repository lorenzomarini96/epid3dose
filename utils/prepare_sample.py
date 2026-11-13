
"""
Prepare a sample CT, EPID and RTDOSE case for epid3dose.

The script:
    1. Loads the CT DICOM series.
    2. Loads the reference RTDOSE.
    3. Resamples CT onto the RTDOSE grid.
    4. Normalizes and crops/pads CT and RTDOSE.
    5. Preprocesses the EPID image.
    6. Saves the three arrays as NumPy files.

Run from the project root:

    python -m utils.prepare_sample
"""

from pathlib import Path

import numpy as np

from .constants import (
    CT_SHAPE,
    EPID_SHAPE,
    DOSE_SHAPE,
)

from .preprocessing import (
    preprocess_ct_rtdose_pair,
    preprocess_epid,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "examples" / "input"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "examples"
    / "output"
    / "preprocessed"
)

SAMPLE_NAME = "CIRS_centro_0gradi_3"

CT_DIR = INPUT_DIR / f"CT_{SAMPLE_NAME}"
EPID_FILE = INPUT_DIR / f"EPID_{SAMPLE_NAME}.dcm"
RTDOSE_FILE = INPUT_DIR / f"RTDOSE_{SAMPLE_NAME}.dcm"


# =============================================================================
# OUTPUT
# =============================================================================

def save_array(array, output_path):
    """Save a NumPy array as float32."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_path,
        np.asarray(array, dtype=np.float32)
    )

    print(f"Saved: {output_path.name}")
    print(f"Shape: {array.shape}")
    print(f"Dtype: {array.dtype}")
    print(
        f"Range: {array.min():.6f} / "
        f"{array.max():.6f}"
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def prepare_sample():

    print("=" * 70)
    print("EPID3DOSE - SAMPLE PREPROCESSING")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Check input files
    # -----------------------------------------------------------------

    if not CT_DIR.is_dir():
        raise FileNotFoundError(
            f"CT directory not found: {CT_DIR}"
        )

    if not EPID_FILE.is_file():
        raise FileNotFoundError(
            f"EPID file not found: {EPID_FILE}"
        )

    if not RTDOSE_FILE.is_file():
        raise FileNotFoundError(
            f"RTDOSE file not found: {RTDOSE_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nInput files:")
    print(f"CT:     {CT_DIR}")
    print(f"EPID:   {EPID_FILE.name}")
    print(f"RTDOSE: {RTDOSE_FILE.name}")

    # -----------------------------------------------------------------
    # CT + RTDOSE preprocessing
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CT AND RTDOSE PREPROCESSING")
    print("=" * 70)

    ct_array, dose_array = preprocess_ct_rtdose_pair(
        ct_folder=CT_DIR,
        dose_path=RTDOSE_FILE,
        target_shape=CT_SHAPE
    )

    if ct_array.shape != CT_SHAPE:
        raise RuntimeError(
            f"Unexpected CT shape: {ct_array.shape}"
        )

    if dose_array.shape != DOSE_SHAPE:
        raise RuntimeError(
            f"Unexpected RTDOSE shape: {dose_array.shape}"
        )

    # -----------------------------------------------------------------
    # EPID preprocessing
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EPID PREPROCESSING")
    print("=" * 70)

    epid_array = preprocess_epid(
        epid_path=EPID_FILE
    )

    expected_epid_shape = (
        EPID_SHAPE[0],
        EPID_SHAPE[1],
        1
    )

    if epid_array.shape != expected_epid_shape:
        raise RuntimeError(
            f"Unexpected EPID shape: {epid_array.shape}"
        )

    # -----------------------------------------------------------------
    # Save arrays
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING PREPROCESSED ARRAYS")
    print("=" * 70)

    save_array(
        ct_array,
        OUTPUT_DIR / f"CT_{SAMPLE_NAME}.npy"
    )

    save_array(
        epid_array,
        OUTPUT_DIR / f"EPID_{SAMPLE_NAME}.npy"
    )

    save_array(
        dose_array,
        OUTPUT_DIR / f"RTDOSE_{SAMPLE_NAME}.npy"
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\nFinal arrays:")
    print(f"CT:     {ct_array.shape}")
    print(f"EPID:   {epid_array.shape}")
    print(f"RTDOSE: {dose_array.shape}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    prepare_sample()
