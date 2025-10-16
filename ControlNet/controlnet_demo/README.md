# 🧠 ControlNet Demo — Modular Refactor for ZEISS ML Engineer Task

This folder contains the **refactored and modularized implementation** of the ControlNet demo (`awesomedemo.py`) as part of the ZEISS Machine Learning Engineer assignment (Task 3).

The refactor transforms the original monolithic prototype into a **configuration-driven, object-oriented, and reproducible codebase** suitable for industrial research workflows.

---

## 🚀 Overview

**Original repository:** [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)  
**Base model:** `control_sd15_canny.pth`  
**Objective:** Generate synthetic MRI brain scans conditioned on Canny edges, demonstrating controllable diffusion-based image synthesis.

## ✨ Key Improvements
- Modular architecture (`data/`, `models/`, `training/`, `inference/`, `utils/`)
- YAML-based configuration and experiment management  
- Structured logging with timestamped run directories  
- Deterministic seeding and environment locking  
- Lightweight dependency footprint  
- Docker-compatible for reproducible execution (`zeiss-controlnet:latest`)

---

## 🧰 Environment Setup

The original `environment.yaml` from the upstream ControlNet repository contained several **outdated or deprecated dependencies** (e.g., mismatched `torch`/`torchvision` versions).  
To ensure smooth execution and long-term reproducibility, a **new environment specification** has been created specifically for this refactored demo.

### The updated file is located at:
```
controlnet_demo/environment.yaml
```

### Create the environment

```bash
cd controlnet_demo
conda env create -f environment.yaml
conda activate controlnet_demo
```

### This environment:

1. Pins compatible versions of torch, torchvision, transformers, and supporting libraries.
2. Is fully compatible with CUDA-based containerization and can be easily integrated into a Docker image if needed (e.g., for future deployment or reproducibility pipelines).
3. Reduces dependency size for faster builds and cleaner reproducibility.

## 📁 Directory Structure
```
controlnet_demo/
├── configs/             # YAML configs for model, training, and inference
├── data/                # Data loading and preprocessing logic
├── models/              # Model factories and architecture definitions
├── training/            # Training orchestration and experiment logging
├── inference/           # Inference and visualization utilities
├── utils/               # Logging, configuration, and reproducibility utilities
├── runs/                # Auto-generated experiment logs and results
├── test_imgs/           # Input test images
├── tests/               # Validation scripts for core modules
├── environment.yaml     # Updated and lightweight environment specification
├── main.py              # Entry point for running the demo
└── README.md            # This file
```

## ⚙️ Running the Demo
### 1. Prerequisites

#### Download the pretrained ControlNet model and place it under:
```
./models/control_sd15_canny.pth
```

#### Activate the environment:
```
conda activate controlnet_demo
```

### 2. Run the Demo
```bash
cd controlnet_demo
python main.py --config configs/mri_canny_sd15.yaml
```
#### All results and logs will be saved automatically under:
```
runs/YYYYMMDD_HHMMSS_mri_canny_sd15/
```

## 🧠 Example Output

Below is an example of a generated synthetic MRI scan conditioned on Canny edges:

![image info](ControlNet/controlnet_demo/runs/20251016_164937_mri_canny_sd15/outputs/result.png)

## 🔁 Reproducibility

* Fixed random seeds for torch, numpy, and Python’s random module.
* PyTorch deterministic mode enforced.
* Uses the updated controlnet_demo/environment.yaml for modernized dependency management.

