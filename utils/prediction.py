
"""
Prediction utilities for epid3dose.

This module provides:
    - Model construction and weight loading
    - Preprocessed CT and EPID loading
    - 3D dose prediction
    - Dose denormalization
    - Prediction saving

Inference workflow:

    CT (128,128,128)
             +
    EPID (256,256,1)
             |
             v
         3D U-Net
             |
             v
    Predicted dose (128,128,128)

The reference RTDOSE is not required for inference.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

import time

from models.unet_3d import build_model

from .report import save_prediction_report, save_excel_report

from .constants import (
    CT_SHAPE,
    EPID_SHAPE,
    DOSE_SHAPE,
    INITIAL_FILTERS,
    OUTPUT_ACTIVATION,
    DOSE_GLOBAL_MIN,
    DOSE_GLOBAL_MAX,
)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_ALPHA = 0.1
MODEL_LIFT_ALPHA = 1.0


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_inputs(ct_array, epid_array):
    """
    Validate CT and EPID arrays before inference.

    Expected shapes:
        CT:   (128,128,128)
        EPID: (256,256,1)
    """

    expected_epid_shape = (
        EPID_SHAPE[0],
        EPID_SHAPE[1],
        1
    )

    if ct_array.shape != CT_SHAPE:
        raise ValueError(
            f"Invalid CT shape: {ct_array.shape}. "
            f"Expected {CT_SHAPE}."
        )

    if epid_array.shape != expected_epid_shape:
        raise ValueError(
            f"Invalid EPID shape: {epid_array.shape}. "
            f"Expected {expected_epid_shape}."
        )

    if not np.all(np.isfinite(ct_array)):
        raise ValueError("CT contains NaN or infinite values.")

    if not np.all(np.isfinite(epid_array)):
        raise ValueError("EPID contains NaN or infinite values.")

    return True


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(
    weights_path,
    base_filters=INITIAL_FILTERS,
    alpha=MODEL_ALPHA,
    lift_alpha=MODEL_LIFT_ALPHA
):
    """
    Build the 3D U-Net and load trained weights.

    Parameters
    ----------
    weights_path : str or Path
        Path to the trained model weights.

    base_filters : int
        Number of initial convolutional filters.

    alpha : float
        LeakyReLU negative slope.

    lift_alpha : float
        Exponential decay parameter for weighted EPID lifting.

    Returns
    -------
    model : tensorflow.keras.Model
        Model with trained weights loaded.
    """

    weights_path = Path(weights_path)

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    print("\n[MODEL]")
    print("Building 3D U-Net...")

    model = build_model(
        base_filters=base_filters,
        alpha=alpha,
        D=CT_SHAPE[0],
        lift_alpha=lift_alpha
    )

    print(f"Output activation: {OUTPUT_ACTIVATION}")
    print(f"Initial filters: {base_filters}")
    print(f"Weighted lift alpha: {lift_alpha}")

    print(f"\nLoading weights: {weights_path.name}")

    model.load_weights(str(weights_path))

    print("Model weights loaded successfully.")

    return model


# =============================================================================
# INPUT LOADING
# =============================================================================


def load_preprocessed_inputs(ct_path, epid_path):
    """
    Load preprocessed CT and EPID NumPy arrays.

    Accepted EPID shapes:
        (256, 256)
        (256, 256, 1)

    Returns:
        CT:   (128, 128, 128)
        EPID: (256, 256, 1)
    """

    ct_path = Path(ct_path)
    epid_path = Path(epid_path)

    if not ct_path.is_file():
        raise FileNotFoundError(
            f"CT array not found: {ct_path}"
        )

    if not epid_path.is_file():
        raise FileNotFoundError(
            f"EPID array not found: {epid_path}"
        )

    ct_array = np.load(ct_path).astype(np.float32)
    epid_array = np.load(epid_path).astype(np.float32)

    # Add channel dimension if missing.
    if epid_array.ndim == 2:
        epid_array = epid_array[..., np.newaxis]

    validate_inputs(ct_array, epid_array)

    print("\n[INPUTS LOADED]")
    print(f"CT shape: {ct_array.shape}")
    print(f"EPID shape: {epid_array.shape}")

    return ct_array, epid_array


# =============================================================================
# DOSE DENORMALIZATION
# =============================================================================

def denormalize_dose(
    dose_normalized,
    global_min=DOSE_GLOBAL_MIN,
    global_max=DOSE_GLOBAL_MAX
):
    """
    Convert normalized dose values back to Gy.

    Inverse normalization:

        Dose_Gy = Dose_normalized * (global_max - global_min)
                  + global_min
    """

    if global_min is None or global_max is None:
        raise ValueError(
            "Dose normalization values are not configured."
        )

    if global_max <= global_min:
        raise ValueError(
            "global_max must be greater than global_min."
        )

    dose_gy = (
        dose_normalized * (global_max - global_min)
        + global_min
    )

    return dose_gy.astype(np.float32)


# =============================================================================
# DOSE PREDICTION
# =============================================================================

def predict_dose(
    model,
    ct_array,
    epid_array,
    batch_size=1,
    verbose=1
):
    """
    Predict a 3D dose volume from CT and EPID.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Loaded 3D U-Net.

    ct_array : np.ndarray
        Preprocessed CT, shape (128,128,128).

    epid_array : np.ndarray
        Preprocessed EPID, shape (256,256,1).

    batch_size : int
        Inference batch size.

    verbose : int
        Keras prediction verbosity.

    Returns
    -------
    dose_normalized : np.ndarray
        Predicted normalized dose, shape (128,128,128).

    dose_gy : np.ndarray
        Predicted dose in Gy, shape (128,128,128).
    """

    validate_inputs(
        ct_array,
        epid_array
    )

    # Add batch and channel dimensions to CT.
    ct_input = ct_array[np.newaxis, ..., np.newaxis]

    # Add batch dimension to EPID.
    epid_input = epid_array[np.newaxis, ...]

    print("\n[PREDICTION]")
    print(f"CT input: {ct_input.shape}")
    print(f"EPID input: {epid_input.shape}")

    # Run model inference.
    prediction = model.predict(
        [ct_input, epid_input],
        batch_size=batch_size,
        verbose=verbose
    )

    # Remove batch and channel dimensions.
    dose_normalized = prediction[0, ..., 0].astype(
        np.float32
    )

    if dose_normalized.shape != DOSE_SHAPE:
        raise RuntimeError(
            f"Unexpected prediction shape: "
            f"{dose_normalized.shape}. "
            f"Expected {DOSE_SHAPE}."
        )

    # Convert normalized output to Gy.
    dose_gy = denormalize_dose(
        dose_normalized
    )

    print("\n[PREDICTION COMPLETED]")
    print(f"Output shape: {dose_normalized.shape}")
    print(
        f"Normalized range: "
        f"{dose_normalized.min():.6f} / "
        f"{dose_normalized.max():.6f}"
    )
    print(
        f"Dose range: "
        f"{dose_gy.min():.5f} / "
        f"{dose_gy.max():.5f} Gy"
    )

    return dose_normalized, dose_gy


# =============================================================================
# PREDICTION SAVING
# =============================================================================

def save_prediction(
    dose_normalized,
    dose_gy,
    output_dir,
    sample_name
):
    """
    Save normalized and physical predicted dose volumes.

    Output files:
        - *_normalized.npy
        - *_Gy.npy
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    normalized_path = (
        output_dir / f"{sample_name}_normalized.npy"
    )

    dose_path = (
        output_dir / f"{sample_name}_Gy.npy"
    )

    np.save(
        normalized_path,
        dose_normalized.astype(np.float32)
    )

    np.save(
        dose_path,
        dose_gy.astype(np.float32)
    )

    print("\n[FILES SAVED]")
    print(normalized_path)
    print(dose_path)


# =============================================================================
# COMPLETE INFERENCE PIPELINE
# =============================================================================


def run_prediction(
    ct_path,
    epid_path,
    weights_path,
    output_dir,
    sample_name="prediction",
    base_filters=16,
    alpha=MODEL_ALPHA,
    lift_alpha=MODEL_LIFT_ALPHA
):
    """
    Complete inference pipeline.

    Outputs:
        - Normalized predicted dose
        - Predicted dose in Gy
        - Single PDF prediction report
    """

    start_time = time.perf_counter()

    print("=" * 70)
    print("EPID3DOSE - 3D DOSE PREDICTION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. LOAD INPUTS
    # -------------------------------------------------------------------------

    ct_array, epid_array = load_preprocessed_inputs(
        ct_path,
        epid_path
    )

    # -------------------------------------------------------------------------
    # 2. LOAD MODEL
    # -------------------------------------------------------------------------

    model = load_model(
        weights_path=weights_path,
        base_filters=base_filters,
        alpha=alpha,
        lift_alpha=lift_alpha
    )

    # -------------------------------------------------------------------------
    # 3. PREDICTION
    # -------------------------------------------------------------------------

    dose_normalized, dose_gy = predict_dose(
        model=model,
        ct_array=ct_array,
        epid_array=epid_array
    )

    # -------------------------------------------------------------------------
    # 4. SAVE PREDICTION
    # -------------------------------------------------------------------------

    save_prediction(
        dose_normalized=dose_normalized,
        dose_gy=dose_gy,
        output_dir=output_dir,
        sample_name=sample_name
    )

    # -------------------------------------------------------------------------
    # 5. MODEL CONFIGURATION
    # -------------------------------------------------------------------------

    model_config = {
        "architecture": "unet_3d",
        "base_filters": base_filters,
        "leaky_relu_alpha": alpha,
        "weighted_lift_alpha": lift_alpha,
        "output_activation": OUTPUT_ACTIVATION,
        "weights_file": Path(weights_path).name,
    }

    elapsed_time = time.perf_counter() - start_time

    # -------------------------------------------------------------------------
    # 6. GENERATE PDF REPORT
    # -------------------------------------------------------------------------

    report_dir = Path(output_dir).parent / "reports"

    save_prediction_report(
        sample_name=sample_name,
        ct_path=ct_path,
        epid_path=epid_path,
        weights_path=weights_path,
        ct_array=ct_array,
        epid_array=epid_array,
        dose_normalized=dose_normalized,
        dose_gy=dose_gy,
        model_config=model_config,
        output_dir=report_dir,
        elapsed_time=elapsed_time,
    )

    save_excel_report(
        sample_name=sample_name,
        ct_path=ct_path,
        epid_path=epid_path,
        weights_path=weights_path,
        ct_array=ct_array,
        epid_array=epid_array,
        dose_normalized=dose_normalized,
        dose_gy=dose_gy,
        model_config=model_config,
        output_dir=report_dir,
        elapsed_time=elapsed_time,
    )

    # -------------------------------------------------------------------------
    # 7. FINISH
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("INFERENCE FINISHED")
    print("=" * 70)

    return dose_normalized, dose_gy
