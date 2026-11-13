# epid3dose
Deep learning framework for 3D in-vivo dose reconstruction from transit EPID images and planning CT.

```{=html}
<!-- Badges -->
```
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

**EPID3DOSE** is an open-source Python package for 3D dose
reconstruction in radiotherapy using deep learning.

The package implements a 3D U-Net-based framework that takes two inputs:

-   A three-dimensional CT volume.
-   A two-dimensional Electronic Portal Imaging Device (EPID) image
    acquired during treatment.

The model combines anatomical information from the CT volume with
transit dosimetric information from the EPID image to predict a
three-dimensional dose distribution.

The main objective is to provide a computational framework for in-vivo
dose reconstruction, supporting the development and evaluation of
EPID-based dosimetric verification methods.

## Workflow

The reconstruction workflow consists of the following main steps:

1.  **Input loading:** loading the preprocessed CT volume and EPID
    image.
2.  **Model initialization:** loading the trained 3D U-Net architecture
    and its weights.
3.  **Dose reconstruction:** processing the CT and EPID inputs to
    predict the 3D dose distribution.
4.  **Dose conversion:** converting the normalized prediction into dose
    values expressed in Gy.
5.  **Report generation:** automatically generating PDF and Excel
    reports containing the input information, model configuration, and
    prediction statistics.

## Model architecture

EPID3DOSE uses an early-fusion 3D U-Net architecture.

The CT volume provides the anatomical information, while the EPID image
is incorporated into the network through an internal lifting mechanism
that allows the two-dimensional image to contribute to the
three-dimensional reconstruction process.

The network produces a volumetric dose prediction with the same spatial
dimensions as the input CT volume.

## Repository structure

``` text
epid3dose/
│
├── docs/
│   ├── images/
│   └── source/
│
├── examples/
│   ├── input/
│   └── output/
│       ├── figures/
│       ├── prediction/
│       ├── preprocessed/
│       └── reports/
│
├── models/
│   ├── weights/
│   │   └── best.weights.h5
│   └── unet_3d.py
│
├── utils/
│   ├── constants.py
│   ├── io.py
│   ├── plotting.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── report.py
│
├── epid3dose.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

Clone the repository:

``` bash
git clone https://github.com/lorenzomarini/epid3dose.git
cd epid3dose
```

Create and activate a virtual environment:

``` bash
python -m venv venv
```

On Linux/macOS:

``` bash
source venv/bin/activate
```

On Windows:

``` bash
venv\Scripts\activate
```

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

## Trained model weights

The trained model weights are provided in:

``` text
models/weights/best.weights.h5
```

The current checkpoint is configured for a 3D U-Net with 16 initial
convolutional filters.

## Example data

An example dataset is provided in the `examples/` directory.

The inference pipeline currently accepts preprocessed NumPy arrays
(`.npy`) containing:

-   CT volume: `(128, 128, 128)`
-   EPID image: `(256, 256)`

The arrays must follow the preprocessing and normalization conventions
used during model training.

## Usage

### 3D dose prediction

Run the prediction from the command line:

``` bash
python epid3dose.py predict \
    --ct examples/output/preprocessed/CT_CIRS_centro_0gradi_3.npy \
    --epid examples/output/preprocessed/EPID_CIRS_centro_0gradi_3.npy \
    --weights models/weights/best.weights.h5 \
    --name CIRS_centro_0gradi_3 \
    --base-filters 16
```

The command loads the CT and EPID arrays, initializes the trained model,
and performs the 3D dose reconstruction.

### Demo notebook

Work in progress...

## Output

For each prediction, EPID3DOSE generates the following outputs.

### Predicted dose volumes

Two NumPy arrays are saved in `examples/output/prediction/`:

-   `<sample_name>_normalized.npy`: normalized predicted dose
    distribution.
-   `<sample_name>_Gy.npy`: predicted dose distribution expressed in Gy.

### Automatic reports

Two reports are automatically generated in `examples/output/reports/`:

-   `<sample_name>_report.pdf`: formatted report containing the input
    information, model configuration, preprocessing information, and
    prediction statistics.
-   `<sample_name>_report.xlsx`: spreadsheet containing the same
    information, organized into separate worksheets.

### Visualization

The `utils/plotting.py` module provides functions for visualizing and
comparing three-dimensional dose distributions.

## Validation

The reconstructed dose distribution can be compared with a reference
dose distribution obtained from the treatment planning system (TPS).

The package includes utilities for loading and visualizing dose volumes,
allowing qualitative and quantitative evaluation of the reconstruction
results.

The predicted dose should be independently validated against the
reference dose before any clinical interpretation or application.

## Requirements

Main dependencies include:

-   Python
-   TensorFlow
-   NumPy
-   SciPy
-   pydicom
-   SimpleITK
-   Matplotlib
-   ReportLab
-   OpenPyXL

See `requirements.txt` for the complete list.

## Citation

If you use EPID3DOSE in your research, please cite the corresponding
publication:

*Citation information to be added.*

## Acknowledgements

This work was developed within the research activities on EPID-based
in-vivo dosimetry and deep learning for radiotherapy dose
reconstruction.

## License

This project is licensed under the MIT License. See the `LICENSE` file
for details.
