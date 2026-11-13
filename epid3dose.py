
"""
epid3dose
---------

3D dose reconstruction from CT and transit EPID images
using a trained 3D U-Net.

Usage
-----

Run a prediction:

    python epid3dose.py predict \
        --ct examples/output/preprocessed/CT_sample.npy \
        --epid examples/output/preprocessed/EPID_sample.npy \
        --weights models/weights/best_model.weights.h5

Specify a custom output directory:

    python epid3dose.py predict \
        --ct path/to/CT.npy \
        --epid path/to/EPID.npy \
        --weights path/to/model.weights.h5 \
        --output examples/output/prediction \
        --name sample_001

Display available commands:

    python epid3dose.py --help
    python epid3dose.py predict --help
"""

import argparse
import sys
from pathlib import Path

from utils.prediction import run_prediction


# =============================================================================
# PACKAGE INFORMATION
# =============================================================================

PACKAGE_NAME = "epid3dose"
VERSION = "0.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "examples" / "output" / "prediction"
)


# =============================================================================
# COMMAND: PREDICT
# =============================================================================

def predict_command(args):
    """
    Execute the 3D dose prediction pipeline.
    """

    print("\n" + "=" * 70)
    print(f"{PACKAGE_NAME.upper()} v{VERSION}")
    print("3D IN-VIVO DOSE RECONSTRUCTION")
    print("=" * 70)

    ct_path = Path(args.ct)
    epid_path = Path(args.epid)
    weights_path = Path(args.weights)
    output_dir = Path(args.output)

    # Check input files.
    required_files = {
        "CT": ct_path,
        "EPID": epid_path,
        "Model weights": weights_path,
    }

    for name, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} file not found: {path}"
            )

    print("\n[CONFIGURATION]")
    print(f"CT:             {ct_path}")
    print(f"EPID:           {epid_path}")
    print(f"Model weights:  {weights_path}")
    print(f"Output folder:  {output_dir}")
    print(f"Sample name:    {args.name}")

    # Run inference.
    dose_normalized, dose_gy = run_prediction(
        ct_path=ct_path,
        epid_path=epid_path,
        weights_path=weights_path,
        output_dir=output_dir,
        sample_name=args.name,
        base_filters=args.base_filters,
        alpha=args.alpha,
        lift_alpha=args.lift_alpha,
    )

    print("\n[SUMMARY]")
    print(f"Predicted volume: {dose_gy.shape}")
    print(
        f"Dose range: "
        f"{dose_gy.min():.5f} - "
        f"{dose_gy.max():.5f} Gy"
    )

    print("\nPrediction completed successfully.")

    return dose_normalized, dose_gy


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_parser():
    """
    Build the command-line interface.
    """

    parser = argparse.ArgumentParser(
        prog=PACKAGE_NAME,
        description=(
            "3D dose reconstruction from CT and transit EPID "
            "using a trained 3D U-Net."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"{PACKAGE_NAME} {VERSION}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="Available commands",
    )

    # -------------------------------------------------------------------------
    # PREDICT
    # -------------------------------------------------------------------------

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict a 3D dose volume from CT and EPID.",
        description=(
            "Run the complete inference pipeline using "
            "preprocessed CT and EPID arrays and trained model weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    predict_parser.add_argument(
        "--ct",
        type=str,
        required=True,
        help="Path to the preprocessed CT NumPy file.",
    )

    predict_parser.add_argument(
        "--epid",
        type=str,
        required=True,
        help="Path to the preprocessed EPID NumPy file.",
    )

    predict_parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained model weights.",
    )

    predict_parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where predictions will be saved.",
    )

    predict_parser.add_argument(
        "--name",
        type=str,
        default="prediction",
        help="Name used for the output files.",
    )

    predict_parser.add_argument(
        "--base-filters",
        type=int,
        default=32,
        help="Number of initial convolutional filters.",
    )

    predict_parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="LeakyReLU negative slope.",
    )

    predict_parser.add_argument(
        "--lift-alpha",
        type=float,
        default=1.0,
        help="Weighted EPID lifting decay parameter.",
    )

    predict_parser.set_defaults(
        func=predict_command
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main entry point.
    """

    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)

    except KeyboardInterrupt:
        print("\nPrediction interrupted by user.")
        sys.exit(130)

    except Exception as error:
        print(
            f"\nERROR: {error}",
            file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
