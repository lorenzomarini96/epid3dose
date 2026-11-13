
"""
Preprocessing utilities for epid3dose.

This module implements the preprocessing pipeline used to prepare
CT, EPID and RTDOSE data for the 3D dose reconstruction model.

Available operations
--------------------
CT:
    - DICOM series loading
    - Hounsfield Unit conversion
    - Resampling onto the RTDOSE reference grid
    - HU clipping and normalization
    - Center crop / padding

RTDOSE:
    - DICOM loading
    - Conversion to Gy
    - Center crop / padding
    - Global normalization

EPID:
    - DICOM loading
    - Detector response correction using PSF
    - Optional central crop
    - Resize to 256 x 256
    - Global normalization

Array conventions
-----------------
3D volumes: (Z, Y, X)
2D images:  (Y, X)
EPID model input: (256, 256, 1)
"""

from pathlib import Path

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk

from .constants import (
    CT_SHAPE,
    CT_HU_MIN,
    CT_HU_MAX,
    EPID_SHAPE,
    EPID_MIN,
    EPID_MAX,
    EPID_PSF_TAG,
    EPID_DETECTOR_PIXEL_SPACING,
    EPID_CROP_PD_SHAPE,
    PD_PIXEL_SPACING,
    DOSE_SHAPE,
    DOSE_GLOBAL_MIN,
    DOSE_GLOBAL_MAX,
)


# =============================================================================
# DICOM GEOMETRY UTILITIES
# =============================================================================

def get_sitk_direction(image_orientation):
    """
    Convert DICOM ImageOrientationPatient into a SimpleITK direction matrix.

    DICOM ImageOrientationPatient contains:
        - First three values: direction of increasing image columns.
        - Last three values: direction of increasing image rows.

    SimpleITK direction matrix columns correspond to the physical
    directions of increasing X, Y and Z indices.
    """

    orientation = np.asarray(
        image_orientation,
        dtype=np.float64
    )

    x_direction = orientation[:3]
    y_direction = orientation[3:]

    z_direction = np.cross(
        x_direction,
        y_direction
    )

    direction = np.column_stack([
        x_direction,
        y_direction,
        z_direction
    ])

    return tuple(direction.flatten())


def get_slice_normal(image_orientation):
    """
    Calculate the normal vector to the image plane.
    """

    orientation = np.asarray(
        image_orientation,
        dtype=np.float64
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]

    normal = np.cross(
        row_direction,
        column_direction
    )

    return normal / np.linalg.norm(normal)


# =============================================================================
# CT LOADING
# =============================================================================

def load_ct_images(ct_folder):
    """
    Load a CT DICOM series from a directory.

    Slices are sorted according to their physical position projected
    onto the normal to the image plane.

    Pixel values are converted to Hounsfield Units using:

        HU = PixelValue * RescaleSlope + RescaleIntercept

    Parameters
    ----------
    ct_folder : str or Path
        Directory containing the CT DICOM slices.

    Returns
    -------
    ct_array : np.ndarray
        CT volume with shape (Z, Y, X), in HU.

    ct_spacing_zyx : tuple
        Voxel spacing in (Z, Y, X), in mm.

    ct_origin : np.ndarray
        Physical origin in (X, Y, Z), in mm.

    ct_orientation : np.ndarray
        ImageOrientationPatient.
    """

    ct_folder = Path(ct_folder)

    if not ct_folder.is_dir():
        raise FileNotFoundError(
            f"CT directory not found: {ct_folder}"
        )

    slices = []

    for file_path in ct_folder.iterdir():

        if not file_path.is_file():
            continue

        if file_path.name.startswith("."):
            continue

        try:
            ds = pydicom.dcmread(str(file_path))

            if getattr(ds, "Modality", None) != "CT":
                continue

            if not hasattr(ds, "ImagePositionPatient"):
                continue

            slices.append(ds)

        except Exception:
            continue

    if not slices:
        raise RuntimeError(
            f"No valid CT DICOM slices found in {ct_folder}"
        )

    reference = slices[0]

    reference_orientation = np.asarray(
        reference.ImageOrientationPatient,
        dtype=np.float64
    )

    reference_pixel_spacing = np.asarray(
        reference.PixelSpacing,
        dtype=np.float64
    )

    normal = get_slice_normal(
        reference_orientation
    )

    # Check consistency of the CT series geometry.
    for ds in slices:

        orientation = np.asarray(
            ds.ImageOrientationPatient,
            dtype=np.float64
        )

        pixel_spacing = np.asarray(
            ds.PixelSpacing,
            dtype=np.float64
        )

        if not np.allclose(
            orientation,
            reference_orientation,
            atol=1e-5
        ):
            raise ValueError(
                "Inconsistent ImageOrientationPatient "
                "within the CT series."
            )

        if not np.allclose(
            pixel_spacing,
            reference_pixel_spacing,
            atol=1e-5
        ):
            raise ValueError(
                "Inconsistent PixelSpacing "
                "within the CT series."
            )

    # Sort slices using their physical position along the slice normal.
    slices.sort(
        key=lambda ds: np.dot(
            np.asarray(ds.ImagePositionPatient, dtype=np.float64),
            normal
        )
    )

    # Convert pixel values to HU.
    ct_array = np.stack([
        ds.pixel_array.astype(np.float32)
        * float(getattr(ds, "RescaleSlope", 1.0))
        + float(getattr(ds, "RescaleIntercept", 0.0))
        for ds in slices
    ]).astype(np.float32)

    # PixelSpacing = [row spacing, column spacing].
    dy = float(reference_pixel_spacing[0])
    dx = float(reference_pixel_spacing[1])

    # Calculate slice spacing from physical positions.
    slice_coordinates = np.asarray([
        np.dot(
            np.asarray(ds.ImagePositionPatient, dtype=np.float64),
            normal
        )
        for ds in slices
    ])

    if len(slice_coordinates) > 1:

        differences = np.diff(slice_coordinates)

        dz = float(np.median(differences))

        if dz <= 0 or not np.allclose(
            differences,
            dz,
            atol=1e-3
        ):
            raise ValueError(
                "Non-uniform or invalid CT slice spacing detected."
            )

    else:

        dz = float(
            getattr(
                reference,
                "SpacingBetweenSlices",
                getattr(reference, "SliceThickness", 1.0)
            )
        )

    ct_spacing_zyx = (dz, dy, dx)

    ct_origin = np.asarray(
        slices[0].ImagePositionPatient,
        dtype=np.float64
    )

    ct_orientation = reference_orientation

    print("\n[CT LOADED]")
    print(f"Directory: {ct_folder.name}")
    print(f"Number of slices: {len(slices)}")
    print(f"Shape (Z,Y,X): {ct_array.shape}")
    print(f"Spacing (Z,Y,X): {ct_spacing_zyx} mm")
    print(f"Origin (X,Y,Z): {ct_origin} mm")

    return (
        ct_array,
        ct_spacing_zyx,
        ct_origin,
        ct_orientation
    )


# =============================================================================
# RTDOSE LOADING
# =============================================================================

def load_rtdose(dose_path):
    """
    Load an RTDOSE DICOM and convert pixel values to Gy.

    Dose conversion:

        Dose_Gy = PixelValue * DoseGridScaling

    The function currently supports uniformly spaced, increasing
    GridFrameOffsetVector values.

    Parameters
    ----------
    dose_path : str or Path
        RTDOSE DICOM file.

    Returns
    -------
    dose_array : np.ndarray
        Dose volume with shape (Z, Y, X), in Gy.

    dose_spacing_zyx : tuple
        Grid spacing in (Z, Y, X), in mm.

    dose_origin : np.ndarray
        Physical origin in (X, Y, Z), in mm.

    dose_orientation : np.ndarray
        ImageOrientationPatient.
    """

    dose_path = Path(dose_path)

    if not dose_path.is_file():
        raise FileNotFoundError(
            f"RTDOSE file not found: {dose_path}"
        )

    ds = pydicom.dcmread(str(dose_path))

    if getattr(ds, "Modality", None) != "RTDOSE":
        raise ValueError(
            f"Not an RTDOSE DICOM: {dose_path}"
        )

    if not hasattr(ds, "GridFrameOffsetVector"):
        raise ValueError(
            "RTDOSE is missing GridFrameOffsetVector."
        )

    # Convert dose values to Gy.
    dose_array = (
        ds.pixel_array.astype(np.float32)
        * float(ds.DoseGridScaling)
    )

    pixel_spacing = np.asarray(
        ds.PixelSpacing,
        dtype=np.float64
    )

    grid_offsets = np.asarray(
        ds.GridFrameOffsetVector,
        dtype=np.float64
    )

    if len(grid_offsets) != dose_array.shape[0]:
        raise ValueError(
            "GridFrameOffsetVector length does not match "
            "the number of RTDOSE frames."
        )

    if len(grid_offsets) > 1:

        offset_differences = np.diff(grid_offsets)

        if not np.allclose(
            offset_differences,
            offset_differences[0],
            atol=1e-4
        ):
            raise ValueError(
                "Non-uniform RTDOSE frame spacing detected."
            )

        if offset_differences[0] <= 0:
            raise ValueError(
                "Descending RTDOSE frame offsets are not supported "
                "by this implementation."
            )

        dz = float(offset_differences[0])

    else:

        dz = float(
            getattr(ds, "SliceThickness", 1.0)
        )

    dy = float(pixel_spacing[0])
    dx = float(pixel_spacing[1])

    dose_spacing_zyx = (dz, dy, dx)

    dose_origin = np.asarray(
        ds.ImagePositionPatient,
        dtype=np.float64
    )

    dose_orientation = np.asarray(
        ds.ImageOrientationPatient,
        dtype=np.float64
    )

    # DICOM permits two GridFrameOffsetVector conventions.
    # For the relative convention, the first offset is zero.
    # For the absolute-Z convention, the first offset equals
    # ImagePositionPatient[2] and the orientation must be axial.
    if not np.isclose(grid_offsets[0], 0.0, atol=1e-5):

        axial_orientation = np.allclose(
            dose_orientation,
            [1, 0, 0, 0, 1, 0],
            atol=1e-5
        )

        if (
            axial_orientation
            and np.isclose(
                grid_offsets[0],
                dose_origin[2],
                atol=1e-3
            )
        ):
            dose_origin[2] = grid_offsets[0]

        else:
            raise ValueError(
                "Unsupported GridFrameOffsetVector convention. "
                "Expected relative offsets starting at zero."
            )

    print("\n[RTDOSE LOADED]")
    print(f"File: {dose_path.name}")
    print(f"Shape (Z,Y,X): {dose_array.shape}")
    print(f"Spacing (Z,Y,X): {dose_spacing_zyx} mm")
    print(f"Origin (X,Y,Z): {dose_origin} mm")
    print(f"Dose units: {ds.DoseUnits}")
    print(f"Dose type: {ds.DoseType}")
    print(f"Dose range: {dose_array.min():.5f} - "
          f"{dose_array.max():.5f} Gy")

    return (
        dose_array,
        dose_spacing_zyx,
        dose_origin,
        dose_orientation
    )


# =============================================================================
# CT RESAMPLING
# =============================================================================

def resample_ct_to_dose(
    ct_array,
    ct_spacing_zyx,
    ct_origin,
    ct_orientation,
    dose_spacing_zyx,
    dose_shape_zyx,
    dose_origin,
    dose_orientation,
    default_pixel_value=-1024.0
):
    """
    Resample CT onto the RTDOSE reference grid.

    Linear interpolation is used.

    Both input volumes follow NumPy convention (Z,Y,X).
    SimpleITK spacing and size are specified in (X,Y,Z).

    The resulting CT has the same physical grid as the RTDOSE.
    """

    # Create CT SimpleITK image.
    ct_image = sitk.GetImageFromArray(
        np.asarray(ct_array, dtype=np.float32)
    )

    dz, dy, dx = ct_spacing_zyx

    ct_image.SetSpacing((
        dx,
        dy,
        dz
    ))

    ct_image.SetOrigin(
        tuple(ct_origin)
    )

    ct_image.SetDirection(
        get_sitk_direction(ct_orientation)
    )

    # Configure the output grid using RTDOSE geometry.
    dz_dose, dy_dose, dx_dose = dose_spacing_zyx

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing((
        dx_dose,
        dy_dose,
        dz_dose
    ))

    resampler.SetSize((
        int(dose_shape_zyx[2]),
        int(dose_shape_zyx[1]),
        int(dose_shape_zyx[0])
    ))

    resampler.SetOutputOrigin(
        tuple(dose_origin)
    )

    resampler.SetOutputDirection(
        get_sitk_direction(dose_orientation)
    )

    resampler.SetDefaultPixelValue(
        float(default_pixel_value)
    )

    resampler.SetInterpolator(
        sitk.sitkLinear
    )

    resampled_image = resampler.Execute(
        ct_image
    )

    ct_resampled = sitk.GetArrayFromImage(
        resampled_image
    ).astype(np.float32)

    if ct_resampled.shape != tuple(dose_shape_zyx):
        raise RuntimeError(
            f"Resampled CT shape {ct_resampled.shape} "
            f"does not match RTDOSE shape {dose_shape_zyx}."
        )

    print("\n[CT RESAMPLED]")
    print(f"Output shape: {ct_resampled.shape}")
    print(
        f"Output range: "
        f"{ct_resampled.min():.2f} / "
        f"{ct_resampled.max():.2f} HU"
    )

    return ct_resampled


# =============================================================================
# CT NORMALIZATION
# =============================================================================

def normalize_ct_hu(
    ct_array,
    hu_min=CT_HU_MIN,
    hu_max=CT_HU_MAX
):
    """
    Clip CT values to the specified HU range and normalize to [0,1].

    Formula:

        CT_normalized = (CT_clipped - HU_min) / (HU_max - HU_min)
    """

    if hu_max <= hu_min:
        raise ValueError(
            "hu_max must be greater than hu_min."
        )

    ct_clipped = np.clip(
        ct_array,
        hu_min,
        hu_max
    )

    ct_normalized = (
        (ct_clipped - hu_min)
        / (hu_max - hu_min)
    )

    return ct_normalized.astype(np.float32)


# =============================================================================
# CENTER CROP / PADDING
# =============================================================================

def center_crop_or_pad(
    volume,
    target_shape=CT_SHAPE,
    pad_value=0.0
):
    """
    Center crop or pad a 3D volume to the requested shape.

    Input and output convention: (Z,Y,X).
    """

    volume = np.asarray(volume)

    target_shape = tuple(target_shape)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume, got shape {volume.shape}."
        )

    # Center crop.
    slices = []

    for current, target in zip(
        volume.shape,
        target_shape
    ):

        if current > target:

            start = (current - target) // 2

            slices.append(
                slice(start, start + target)
            )

        else:

            slices.append(
                slice(0, current)
            )

    volume = volume[tuple(slices)]

    # Center padding.
    padding = []

    for current, target in zip(
        volume.shape,
        target_shape
    ):

        if current < target:

            difference = target - current

            before = difference // 2
            after = difference - before

            padding.append((
                before,
                after
            ))

        else:

            padding.append((
                0,
                0
            ))

    volume = np.pad(
        volume,
        padding,
        mode="constant",
        constant_values=pad_value
    )

    if volume.shape != target_shape:
        raise RuntimeError(
            f"Output shape {volume.shape} "
            f"does not match target {target_shape}."
        )

    return volume.astype(np.float32)


# =============================================================================
# RTDOSE NORMALIZATION
# =============================================================================

def normalize_dose_global(
    dose_array,
    global_min=DOSE_GLOBAL_MIN,
    global_max=DOSE_GLOBAL_MAX
):
    """
    Normalize RTDOSE using fixed global min/max values from training.

    The same normalization parameters must be used during training
    and inference.
    """

    if global_min is None or global_max is None:
        raise ValueError(
            "Global dose normalization values are not configured. "
            "Set DOSE_GLOBAL_MIN and DOSE_GLOBAL_MAX in constants.py."
        )

    if global_max <= global_min:
        raise ValueError(
            "global_max must be greater than global_min."
        )

    dose_normalized = (
        (dose_array - global_min)
        / (global_max - global_min)
    )

    return dose_normalized.astype(np.float32)


# =============================================================================
# EPID CROP
# =============================================================================

def crop_epid_to_match_pd(
    epid_image,
    pd_shape=EPID_CROP_PD_SHAPE,
    pixel_spacing_pd=PD_PIXEL_SPACING,
    pixel_spacing_epid=EPID_DETECTOR_PIXEL_SPACING
):
    """
    Center crop an EPID image to match the physical field of view
    of the original Portal Dose image.

    The crop is calculated using the native detector pixel spacing.

    Parameters
    ----------
    epid_image : np.ndarray
        Original EPID image.

    pd_shape : tuple
        Reference Portal Dose dimensions.

    pixel_spacing_pd : float
        Portal Dose pixel spacing in mm/pixel.

    pixel_spacing_epid : float
        Native EPID detector pixel spacing in mm/pixel.

    Returns
    -------
    cropped : np.ndarray
        Center-cropped EPID image.
    """

    target_physical_x = (
        pd_shape[1] * pixel_spacing_pd
    )

    target_physical_y = (
        pd_shape[0] * pixel_spacing_pd
    )

    target_size_x = int(
        target_physical_x / pixel_spacing_epid
    )

    target_size_y = int(
        target_physical_y / pixel_spacing_epid
    )

    original_y, original_x = epid_image.shape

    if (
        target_size_x > original_x
        or target_size_y > original_y
    ):
        raise ValueError(
            "Requested crop is larger than the original EPID image."
        )

    crop_x = (original_x - target_size_x) // 2
    crop_y = (original_y - target_size_y) // 2

    cropped = epid_image[
        crop_y:crop_y + target_size_y,
        crop_x:crop_x + target_size_x
    ]

    return cropped


# =============================================================================
# EPID PREPROCESSING
# =============================================================================

def preprocess_epid(
    epid_path,
    epid_min=EPID_MIN,
    epid_max=EPID_MAX,
    apply_crop=True,
    pd_shape=EPID_CROP_PD_SHAPE,
    pixel_spacing_pd=PD_PIXEL_SPACING,
    pixel_spacing_epid=EPID_DETECTOR_PIXEL_SPACING
):
    """
    Preprocess one transit EPID DICOM image.

    Processing steps
    ----------------
    1. Read DICOM pixel array.
    2. Apply detector response correction using PSF.
    3. Optionally crop the image to match the PD field of view.
    4. Resize to 256 x 256 using INTER_AREA.
    5. Normalize using fixed EPID_MIN and EPID_MAX values.

    No additional clipping is applied after normalization.

    Returns
    -------
    epid_normalized : np.ndarray
        Normalized EPID image with shape (256,256,1).
    """

    epid_path = Path(epid_path)

    if not epid_path.is_file():
        raise FileNotFoundError(
            f"EPID file not found: {epid_path}"
        )

    if epid_min is None or epid_max is None:
        raise ValueError(
            "EPID normalization values are not configured."
        )

    if epid_max <= epid_min:
        raise ValueError(
            "epid_max must be greater than epid_min."
        )

    ds = pydicom.dcmread(
        str(epid_path)
    )

    if getattr(ds, "Modality", None) != "RTIMAGE":
        raise ValueError(
            f"Expected RTIMAGE DICOM, got "
            f"{getattr(ds, 'Modality', None)}."
        )

    if EPID_PSF_TAG not in ds:
        raise ValueError(
            f"Missing PSF private tag {EPID_PSF_TAG} "
            f"in {epid_path.name}."
        )

    raw_image = ds.pixel_array.astype(
        np.float32
    )

    psf_value = float(
        ds[EPID_PSF_TAG].value
    )

    if not np.isfinite(psf_value) or psf_value <= 0:
        raise ValueError(
            f"Invalid PSF value in {epid_path.name}: {psf_value}"
        )

    # 1. Detector response correction.
    epid_corrected = (
        65535.0 - raw_image
    ) / psf_value

    # 2. Optional central crop.
    if apply_crop:

        epid_corrected = crop_epid_to_match_pd(
            epid_corrected,
            pd_shape=pd_shape,
            pixel_spacing_pd=pixel_spacing_pd,
            pixel_spacing_epid=pixel_spacing_epid
        )

    # 3. Resize to the model input dimensions.
    epid_resized = cv2.resize(
        epid_corrected.astype(np.float32),
        (EPID_SHAPE[1], EPID_SHAPE[0]),
        interpolation=cv2.INTER_AREA
    )

    # 4. Global normalization using training statistics.
    epid_normalized = (
        (epid_resized - epid_min)
        / (epid_max - epid_min)
    ).astype(np.float32)

    # Add channel dimension.
    epid_normalized = epid_normalized[..., np.newaxis]

    print("\n[EPID PREPROCESSED]")
    print(f"File: {epid_path.name}")
    print(f"Output shape: {epid_normalized.shape}")
    print(
        f"Normalized range: "
        f"{epid_normalized.min():.5f} / "
        f"{epid_normalized.max():.5f}"
    )

    return epid_normalized


# =============================================================================
# COMPLETE CT + RTDOSE PREPROCESSING
# =============================================================================

def preprocess_ct_rtdose_pair(
    ct_folder,
    dose_path,
    global_dose_min=DOSE_GLOBAL_MIN,
    global_dose_max=DOSE_GLOBAL_MAX,
    target_shape=CT_SHAPE
):
    """
    Prepare one CT/RTDOSE pair for dataset construction and validation.

    Workflow
    --------
    1. Load CT and convert to HU.
    2. Load RTDOSE and convert to Gy.
    3. Resample CT onto the RTDOSE reference grid.
    4. Normalize CT.
    5. Center crop/pad CT.
    6. Center crop/pad RTDOSE.
    7. Normalize RTDOSE using fixed global training values.

    Returns
    -------
    ct_final : np.ndarray
        Normalized CT, shape target_shape.

    dose_final : np.ndarray
        Normalized RTDOSE, shape target_shape.
    """

    if global_dose_min is None or global_dose_max is None:
        raise ValueError(
            "Global dose normalization values are missing. "
            "Update DOSE_GLOBAL_MIN and DOSE_GLOBAL_MAX "
            "in utils/constants.py before processing."
        )

    # -----------------------------------------------------------------
    # Load CT
    # -----------------------------------------------------------------

    (
        ct_array,
        ct_spacing,
        ct_origin,
        ct_orientation
    ) = load_ct_images(ct_folder)

    # -----------------------------------------------------------------
    # Load RTDOSE
    # -----------------------------------------------------------------

    (
        dose_array,
        dose_spacing,
        dose_origin,
        dose_orientation
    ) = load_rtdose(dose_path)

    # -----------------------------------------------------------------
    # Resample CT onto RTDOSE grid
    # -----------------------------------------------------------------

    ct_resampled = resample_ct_to_dose(
        ct_array=ct_array,
        ct_spacing_zyx=ct_spacing,
        ct_origin=ct_origin,
        ct_orientation=ct_orientation,
        dose_spacing_zyx=dose_spacing,
        dose_shape_zyx=dose_array.shape,
        dose_origin=dose_origin,
        dose_orientation=dose_orientation
    )

    # -----------------------------------------------------------------
    # CT normalization and crop/padding
    # -----------------------------------------------------------------

    ct_normalized = normalize_ct_hu(
        ct_resampled
    )

    ct_final = center_crop_or_pad(
        ct_normalized,
        target_shape=target_shape,
        pad_value=0.0
    )

    # -----------------------------------------------------------------
    # RTDOSE crop/padding and global normalization
    # -----------------------------------------------------------------

    dose_cropped = center_crop_or_pad(
        dose_array,
        target_shape=target_shape,
        pad_value=global_dose_min
    )

    dose_final = normalize_dose_global(
        dose_cropped,
        global_min=global_dose_min,
        global_max=global_dose_max
    )

    print("\n[PREPROCESSING COMPLETED]")
    print(f"CT shape: {ct_final.shape}")
    print(f"RTDOSE shape: {dose_final.shape}")

    return ct_final, dose_final
