
"""
Plotting utilities for epid3dose.

Provides visualization tools for comparing:
    - CT volume
    - Reference RTDOSE volume
    - Predicted RTDOSE volume
    - Dose difference

All volumes are expected to have shape (128, 128, 128),
with array ordering (z, y, x).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm

from .constants import (
    DOSE_GLOBAL_MAX,
    CT_SHAPE,
    DOSE_SHAPE,
)


# =============================================================================
# ARRAY LOADING AND VALIDATION
# =============================================================================

def load_volume(path):
    """Load a 3D NumPy volume."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Volume not found: {path}")

    volume = np.load(path).astype(np.float32)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, received shape {volume.shape}."
        )

    if not np.all(np.isfinite(volume)):
        raise ValueError(f"Volume contains NaN or infinite values: {path}")

    return volume


def validate_volumes(ct, dose_reference, dose_prediction):
    """Check that all volumes have compatible dimensions."""

    if ct.shape != CT_SHAPE:
        raise ValueError(
            f"Invalid CT shape: {ct.shape}. Expected {CT_SHAPE}."
        )

    if dose_reference.shape != DOSE_SHAPE:
        raise ValueError(
            f"Invalid reference dose shape: {dose_reference.shape}. "
            f"Expected {DOSE_SHAPE}."
        )

    if dose_prediction.shape != DOSE_SHAPE:
        raise ValueError(
            f"Invalid predicted dose shape: {dose_prediction.shape}. "
            f"Expected {DOSE_SHAPE}."
        )

    if not (
        ct.shape == dose_reference.shape == dose_prediction.shape
    ):
        raise ValueError(
            "CT, reference dose and predicted dose must have "
            "the same spatial dimensions."
        )


# =============================================================================
# SLICE EXTRACTION
# =============================================================================

def get_orthogonal_slices(volume, indices):
    """
    Extract axial, coronal and sagittal slices.

    Array convention:
        volume[z, y, x]

    Parameters
    ----------
    volume : np.ndarray
        3D volume.

    indices : dict
        Slice indices with keys:
            axial
            coronal
            sagittal
    """

    z = indices["axial"]
    y = indices["coronal"]
    x = indices["sagittal"]

    return {
        "Axial": volume[z, :, :],
        "Coronal": volume[:, y, :],
        "Sagittal": volume[:, :, x],
    }


# =============================================================================
# DOSE COMPARISON PLOT
# =============================================================================



def plot_dose_comparison(
    ct,
    dose_reference,
    dose_prediction,
    output_path=None,
    sample_name="Dose comparison",
    slice_indices=None,
    figsize=(17, 12),
    dpi=200,
    show=True,
):
    """
    Compare CT, reference dose, predicted dose and dose difference.

    Each individual slice uses its own min-max color scale.
    All displayed slices are rotated by 180 degrees.
    """

    validate_volumes(ct, dose_reference, dose_prediction)

    if slice_indices is None:
        slice_indices = {
            "axial": ct.shape[0] // 2,
            "coronal": ct.shape[1] // 2,
            "sagittal": ct.shape[2] // 2,
        }

    # Convert reference dose from normalized units to Gy.
    dose_reference = dose_reference * DOSE_GLOBAL_MAX

    # Dose difference.
    dose_difference = dose_prediction - dose_reference

    # Extract slices.
    ct_slices = get_orthogonal_slices(ct, slice_indices)
    reference_slices = get_orthogonal_slices(
        dose_reference, slice_indices
    )
    prediction_slices = get_orthogonal_slices(
        dose_prediction, slice_indices
    )
    difference_slices = get_orthogonal_slices(
        dose_difference, slice_indices
    )

    planes = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(
        3, 4,
        figsize=figsize,
        constrained_layout=True
    )

    fig.suptitle(
        f"3D Dose Reconstruction - {sample_name}",
        fontsize=16,
        fontweight="bold"
    )

    for row, plane in enumerate(planes):

        # Rotate all images consistently.
        ct_slice = np.rot90(ct_slices[plane], 2)
        ref_slice = np.rot90(reference_slices[plane], 2)
        pred_slice = np.rot90(prediction_slices[plane], 2)
        diff_slice = np.rot90(difference_slices[plane], 2)

        # -------------------------------------------------------------
        # CT: individual slice normalization
        # -------------------------------------------------------------

        ct_min = float(np.min(ct_slice))
        ct_max = float(np.max(ct_slice))

        if ct_max <= ct_min:
            ct_max = ct_min + 1e-6

        im_ct = axes[row, 0].imshow(
            ct_slice,
            cmap="gray",
            origin="lower",
            vmin=ct_min,
            vmax=ct_max,
            aspect="equal"
        )

        axes[row, 0].set_title("CT")

        # -------------------------------------------------------------
        # REFERENCE DOSE: individual slice normalization
        # -------------------------------------------------------------

        ref_min = float(np.min(ref_slice))
        ref_max = float(np.max(ref_slice))

        if ref_max <= ref_min:
            ref_max = ref_min + 1e-6

        im_ref = axes[row, 1].imshow(
            ref_slice,
            cmap="inferno",
            origin="lower",
            vmin=ref_min,
            vmax=ref_max,
            aspect="equal"
        )

        axes[row, 1].set_title("Reference dose")

        # -------------------------------------------------------------
        # PREDICTED DOSE: individual slice normalization
        # -------------------------------------------------------------

        pred_min = float(np.min(pred_slice))
        pred_max = float(np.max(pred_slice))

        if pred_max <= pred_min:
            pred_max = pred_min + 1e-6

        im_pred = axes[row, 2].imshow(
            pred_slice,
            cmap="inferno",
            origin="lower",
            vmin=pred_min,
            vmax=pred_max,
            aspect="equal"
        )

        axes[row, 2].set_title("Predicted dose")

        # -------------------------------------------------------------
        # DOSE DIFFERENCE: symmetric scale for each slice
        # -------------------------------------------------------------

        diff_max = float(np.max(np.abs(diff_slice)))

        if diff_max <= 0:
            diff_max = 1e-6

        im_diff = axes[row, 3].imshow(
            diff_slice,
            cmap="coolwarm",
            origin="lower",
            vmin=-diff_max,
            vmax=diff_max,
            aspect="equal"
        )

        axes[row, 3].set_title("Prediction - Reference")

        # Row label.
        axes[row, 0].set_ylabel(
            plane,
            fontsize=12,
            fontweight="bold"
        )

        # Individual colorbars for each slice.
        fig.colorbar(
            im_ct,
            ax=axes[row, 0],
            fraction=0.046,
            pad=0.04
        )

        fig.colorbar(
            im_ref,
            ax=axes[row, 1],
            fraction=0.046,
            pad=0.04,
            label="Gy"
        )

        fig.colorbar(
            im_pred,
            ax=axes[row, 2],
            fraction=0.046,
            pad=0.04,
            label="Gy"
        )

        fig.colorbar(
            im_diff,
            ax=axes[row, 3],
            fraction=0.046,
            pad=0.04,
            label="Gy"
        )

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    if output_path is not None:

        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight"
        )

        print(f"\nFigure saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig



# =============================================================================
# COMPLETE COMPARISON PIPELINE
# =============================================================================

def compare_dose_volumes(
    ct_path,
    reference_path,
    prediction_path,
    output_path,
    sample_name="sample",
    slice_indices=None,
    show=True,
):
    """
    Load CT, reference dose and predicted dose and generate comparison plot.

    All input volumes must be preprocessed and have shape (128,128,128).
    Reference and predicted dose must be expressed in Gy.
    """

    print("\n[PLOTTING]")

    ct = load_volume(ct_path)
    dose_reference = load_volume(reference_path)
    dose_prediction = load_volume(prediction_path)
    dose_reference = dose_reference * DOSE_GLOBAL_MAX
    
    print(f"CT shape: {ct.shape}")
    print(f"Reference dose shape: {dose_reference.shape}")
    print(f"Predicted dose shape: {dose_prediction.shape}")

    # Basic dose statistics.
    print("\n[DOSE STATISTICS]")

    print(
        f"Reference: "
        f"{dose_reference.min():.4f} - "
        f"{dose_reference.max():.4f} Gy"
    )

    print(
        f"Prediction: "
        f"{dose_prediction.min():.4f} - "
        f"{dose_prediction.max():.4f} Gy"
    )

    difference = dose_prediction - dose_reference

    print(
        f"Difference: "
        f"{difference.min():.4f} - "
        f"{difference.max():.4f} Gy"
    )

    return plot_dose_comparison(
        ct=ct,
        dose_reference=dose_reference,
        dose_prediction=dose_prediction,
        output_path=output_path,
        sample_name=sample_name,
        slice_indices=slice_indices,
        show=show,
    )
