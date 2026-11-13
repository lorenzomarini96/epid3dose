
from pathlib import Path

import pydicom
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "input"

CT_DIR = DATA_DIR / "CT_CIRS_centro_0gradi_3"
EPID_FILE = DATA_DIR / "EPID_CIRS_centro_0gradi_3.dcm"
RTDOSE_FILE = DATA_DIR / "RTDOSE_CIRS_centro_0gradi_3.dcm"


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def get_value(dataset, tag, default="Not available"):
    """Safely retrieve a DICOM attribute."""
    return getattr(dataset, tag, default)


def print_geometry(dataset):
    """Print the main spatial geometry attributes."""

    attributes = [
        "Rows",
        "Columns",
        "NumberOfFrames",
        "PixelSpacing",
        "SliceThickness",
        "SpacingBetweenSlices",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "FrameOfReferenceUID",
    ]

    for attribute in attributes:
        value = get_value(dataset, attribute)
        print(f"  {attribute}: {value}")


def read_dicom_header(path):
    """Read DICOM metadata without loading pixel data."""
    return pydicom.dcmread(path, stop_before_pixels=True)


# ---------------------------------------------------------------------
# CT inspection
# ---------------------------------------------------------------------

def inspect_ct():

    print("\n" + "=" * 70)
    print("CT INSPECTION")
    print("=" * 70)

    files = sorted(CT_DIR.glob("*"))

    slices = []

    for file in files:
        if not file.is_file():
            continue

        try:
            ds = read_dicom_header(file)

            if get_value(ds, "Modality") == "CT":
                slices.append((file, ds))

        except Exception:
            continue

    if not slices:
        print("No CT DICOM files found.")
        return

    print(f"\nNumber of CT slices: {len(slices)}")

    first_file, first_ds = slices[0]

    print("\nReference slice:")
    print(f"  File: {first_file.name}")
    print(f"  Modality: {get_value(first_ds, 'Modality')}")

    print_geometry(first_ds)

    # Sort slices using their position along the slice normal.
    orientation = np.asarray(
        first_ds.ImageOrientationPatient, dtype=float
    )

    row_direction = orientation[:3]
    column_direction = orientation[3:]

    slice_normal = np.cross(row_direction, column_direction)

    positions = []

    for file, ds in slices:
        position = np.asarray(ds.ImagePositionPatient, dtype=float)
        slice_coordinate = np.dot(position, slice_normal)
        positions.append((slice_coordinate, file.name))

    positions.sort(key=lambda item: item[0])

    coordinates = np.array([item[0] for item in positions])

    print("\nSlice positions:")
    print(f"  First: {coordinates[0]:.3f} mm")
    print(f"  Last:  {coordinates[-1]:.3f} mm")

    if len(coordinates) > 1:
        differences = np.diff(coordinates)

        print(f"  Mean spacing: {np.mean(np.abs(differences)):.3f} mm")
        print(f"  Minimum spacing: {np.min(np.abs(differences)):.3f} mm")
        print(f"  Maximum spacing: {np.max(np.abs(differences)):.3f} mm")

        if not np.allclose(
            np.abs(differences),
            np.abs(differences[0]),
            atol=1e-3
        ):
            print("  WARNING: Slice spacing is not uniform.")

    print("\nFirst five slices in anatomical order:")
    for coordinate, filename in positions[:5]:
        print(f"  {filename}: {coordinate:.3f} mm")


# ---------------------------------------------------------------------
# EPID inspection
# ---------------------------------------------------------------------

def inspect_epid():

    print("\n" + "=" * 70)
    print("EPID INSPECTION")
    print("=" * 70)

    ds = read_dicom_header(EPID_FILE)

    print(f"\nFile: {EPID_FILE.name}")
    print(f"  Modality: {get_value(ds, 'Modality')}")
    print(f"  SOP Class UID: {get_value(ds, 'SOPClassUID')}")

    print_geometry(ds)

    print("\nAdditional attributes:")
    for attribute in [
        "Exposure",
        "ExposureTime",
        "KVP",
        "ImageType",
        "PhotometricInterpretation",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
    ]:
        print(f"  {attribute}: {get_value(ds, attribute)}")


# ---------------------------------------------------------------------
# RTDOSE inspection
# ---------------------------------------------------------------------

def inspect_rtdose():

    print("\n" + "=" * 70)
    print("RTDOSE INSPECTION")
    print("=" * 70)

    ds = read_dicom_header(RTDOSE_FILE)

    print(f"\nFile: {RTDOSE_FILE.name}")
    print(f"  Modality: {get_value(ds, 'Modality')}")
    print(f"  SOP Class UID: {get_value(ds, 'SOPClassUID')}")

    print_geometry(ds)

    print("\nDosimetric attributes:")
    for attribute in [
        "DoseUnits",
        "DoseType",
        "DoseSummationType",
        "DoseGridScaling",
        "GridFrameOffsetVector",
    ]:
        print(f"  {attribute}: {get_value(ds, attribute)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    inspect_ct()
    inspect_epid()
    inspect_rtdose()

