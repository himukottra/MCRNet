# MCR-Net: Reactant-Only Reaction-Center Localization with Multi-Channel Spatial Molecular Tensors

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2D%20Chemistry-008080.svg)](https://www.rdkit.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MCR-Net** formulates chemical reaction-center localization as a dense 2D spatial prediction task directly from **reactant-only** molecular representations. Instead of relying solely on SMILES sequences or graph-element labels, each molecule is projected onto a shared `64 x 64` grid as a **7-channel Multi-Channel Representation (MCR)** tensor and processed by a 4-level U-Net encoder–decoder architecture.

---

## Key Features & Method Overview

* **7-Channel Spatial Molecular Tensor (`64 x 64`):** Encodes atomic and structural properties on a spatial grid (nominal resolution ~`0.5 Å` per pixel, 2D coordinates generated via [RDKit](https://www.rdkit.org/) `Compute2DCoords`, centered/scaled to ~76% of the grid with a 12% margin):
  * `CH0`: Atomic identity (`σ_a = 1.8` px Gaussian fields)
  * `CH1`: Bond order (`σ_b = 1.2` px line-segment Gaussian fields; single=1, double=2, triple=3, aromatic/other=1.5)
  * `CH2`: Formal charge
  * `CH3`: Aromaticity
  * `CH4`: Ring membership / chemical property encoding
  * `CH5`: Hybridization / hydrogen-count encoding (`sp` / `sp²` / `sp³`)
  * `CH6`: Leakage-control / supervision channel (zeroed out during inference/evaluation via `rc_atoms = None`)
* **U-Net Localization Backbone (~7.8M parameters):** 4-level encoder with `3x3` convolutions, BatchNorm, ReLU, and `2x2` max-pooling, paired with bilinear upsampling, skip connections, and a `1x1` sigmoid output head.
* **Multi-Term Loss & Thresholding:** Trained with spatial mask loss (`L_mask`) plus auxiliary/regularization terms (`L_recon`, `L_delta`, `L_sparse`) using validation-tuned weights `λ = (1.0, 0.1, 0.05, 0.05)` and a validation-selected decision threshold of `0.35`.
* **Leakage Controls:** Evaluates reactant-only inputs without product information or target leakage in `CH6`, alongside a `CH6`-only sanity check.

---

## Repository Structure

```text
MCR-Net/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── tensor_64x64.yaml
│   ├── train_unet.yaml
│   └── eval_uspto50k.yaml
├── src/
│   ├── data/
│   │   ├── rdkit_coords.py
│   │   ├── mcr_tensor.py
│   │   └── ground_truth.py
│   ├── models/
│   │   ├── unet.py
│   │   └── losses.py
│   ├── training/
│   │   ├── train.py
│   │   └── threshold.py
│   └── evaluation/
│       ├── metrics.py
│       └── leakage_checks.py
├── scripts/
│   ├── build_tensors.py
│   ├── run_train.py
│   └── run_eval.py
├── paper/
│   ├── main.tex
│   ├── figures/
│   └── references.bib
├── notebooks/
│   └── tensor_sanity_check.ipynb
└── tests/
    ├── test_tensor_shapes.py
    └── test_leakage_control.py
