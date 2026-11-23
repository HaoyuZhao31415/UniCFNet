# UniCFNet: Burst Image Layer Separation

**A Unified Cardinal Spline Interpolation Network with Frequency-Aware Optimization for Layer Separation**

**Publication status:** The manuscript is being prepared for submission to **Neurocomputing**.

This repository accompanies our work on **UniCFNet**, a unified framework for separating and reconstructing the **base layer** and **overlay layer** from burst image sequences. It retains its original repository name, **FMS-Net**.

UniCFNet addresses five tasks: **reflection removal, occlusion removal, shadow removal, image dehazing, and video motion segmentation**. Its goal is to recover both layers, including useful information in reflections and other overlays.

> **Implementation status:** This README describes the UniCFNet manuscript. The current public `train.py` implements the underlying dual-plane neural-field framework with fixed Hermite interpolation and a photometric objective with optional alpha regularization. The manuscript's spatially adaptive Cardinal tension and frequency-aware loss are not yet integrated into this training entry point. The commands below describe the current implementation; the tables report results from the manuscript.

## Method Overview

UniCFNet fits an implicit representation to **each input burst sequence**. A shared formulation supports the five tasks; this does not imply a single pretrained checkpoint that processes new scenes without optimization.

The method consists of three parts:

1. **Multi-resolution hash-grid representation.** Normalized image coordinates are encoded into spatial features, which support compact neural representations of the base and overlay layers.
2. **Spatially adaptive Cardinal interpolation.** A dual-plane projection model accounts for camera motion and layer depth. For each layer, the spatially adaptive continuous operator (SACO) predicts motion control points and a local, learnable tension parameter, enabling continuous pixel trajectories that vary across the scene.
3. **Frequency-aware optimization.** The decoded base and overlay colors are combined through an alpha matte. A relative photometric loss and a frequency-domain consistency loss compare the composite reconstruction with the observed images.

The compositing model is:

```text
I_hat(t) = (1 - alpha(t)) * C_base(t) + alpha(t) * C_overlay(t)
```

The manuscript's optimization objective is:

```text
L_total = L_rgb + lambda_fft * L_fft
L_rgb   = mean(abs((I_hat - I) / (stop_gradient(I_hat) + epsilon)))
L_fft   = mean(abs(W * (FFT2(I_hat) - FFT2(I))))
```

Here, `W` weights frequency components. The frequency constraint acts on the **composite reconstruction and the observation**. The adaptive tension is spatially varying and is constrained to `[0, 1]` by a sigmoid.

## Results Reported in the Manuscript

The manuscript evaluates the method on the NSF dataset, covering 40 real and synthetic scenes across the five tasks. Quantitative evaluation uses scenes with reference images; scenes without references are evaluated qualitatively.

### Reflection and Occlusion Removal

| Method | Reflection PSNR ↑ | Reflection SSIM ↑ | Occlusion PSNR ↑ | Occlusion SSIM ↑ |
| :--- | ---: | ---: | ---: | ---: |
| DSRNet | 13.2652 | 0.3699 | 13.5131 | 0.3561 |
| NIR | 12.96 | 0.3488 | 14.51 | 0.3364 |
| NSF | 35.49 | 0.8359 | 33.66 | 0.8186 |
| **UniCFNet (ours)** | **38.19** | **0.8829** | **35.38** | **0.8650** |

### Ablation Study

| Variant | PSNR ↑ | SSIM ↑ |
| :--- | ---: | ---: |
| Without SACO | 34.72 | 0.8113 |
| Without adaptive tension | 36.53 | 0.8527 |
| Without frequency-aware optimization | 35.03 | 0.8209 |
| **Full model** | **36.78** | **0.8703** |

These values are transcribed from the manuscript and are not a reproduction using the current public training scripts. Reproducing the tables requires the corresponding method implementation, experiment configurations, ground-truth scene selection, and layer-evaluation protocol.

## Repository Structure

```text
FMS-Net/
├── config/                 # Hash-grid and MLP configurations
├── scripts/                # Existing task-specific training launchers
├── utils/
│   └── utils.py            # RAW processing, interpolation, and helper functions
├── train.py                # Dataset, model, training, and output generation
├── test.py                 # Existing image-comparison script
├── testdtd.py              # Existing image-comparison script
├── tutorial.ipynb          # Interactive notebook
├── requirements.txt        # Dependency specifications
├── data/                   # Suggested location for downloaded data
├── checkpoints/            # Saved checkpoints and scene metadata
└── outputs/                # Reconstructed images and flow fields
```

Training creates a `lightning_logs/` directory for TensorBoard logs.

## Environment Setup

The original code targets **Ubuntu 22.04**, **PyTorch 2.0**, and an **NVIDIA GPU with CUDA**. The requirements file specifies `torch==2.0.0+cu118`; the model also depends on `tinycudann`.

```bash
git clone https://github.com/HaoyuZhao31415/FMS-Net.git
cd FMS-Net

python -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118

# Additional packages imported or used by the current training script
python -m pip install openpyxl scikit-image tensorboard
```

Install the PyTorch bindings for **tiny-cuda-nn** using its [official installation instructions](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension). On a compatible CUDA development environment, the upstream installation command is:

```bash
python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

The requirements file contains several minimum-version constraints and is **not a fully locked environment**. Ensure that the selected Python, PyTorch, PyTorch Lightning, CUDA toolkit, and tiny-cuda-nn versions are compatible. Training uses CUDA-specific components.

## Data Preparation

We use the burst data released with [Neural Spline Fields (NSF)](https://github.com/princeton-computational-imaging/NSF). Download and extract the relevant archives under `data/`, or use another location and pass the path through `--bundle_path`.

| Data | Download | Existing launcher |
| :--- | :--- | :--- |
| Main occlusion scenes | [occlusion-main.zip](https://soap.cs.princeton.edu/nsf/data/occlusion-main.zip) | `scripts/occlusion.sh` |
| Additional occlusion scenes | [occlusion-supp.zip](https://soap.cs.princeton.edu/nsf/data/occlusion-supp.zip) | `scripts/occlusion.sh` |
| In-the-wild occlusion scenes | [occlusion-wild.zip](https://soap.cs.princeton.edu/nsf/data/occlusion-wild.zip) | `scripts/occlusion-wild.sh` |
| Main reflection scenes | [reflection-main.zip](https://soap.cs.princeton.edu/nsf/data/reflection-main.zip) | `scripts/reflection.sh` |
| Additional reflection scenes | [reflection-supp.zip](https://soap.cs.princeton.edu/nsf/data/reflection-supp.zip) | `scripts/reflection.sh` |
| In-the-wild reflection scenes | [reflection-wild.zip](https://soap.cs.princeton.edu/nsf/data/reflection-wild.zip) | `scripts/reflection-wild.sh` |
| Dehazing, shadow, and motion-segmentation scenes | [extras.zip](https://soap.cs.princeton.edu/nsf/data/extras.zip) | Corresponding task launcher below |
| Synthetic validation scenes | [synthetic-validation.zip](https://soap.cs.princeton.edu/nsf/data/synthetic-validation.zip) | Reflection or occlusion launcher with `--rgb_data` |

The training entry point expects a **scene bundle in `.npz` format**, rather than a directory of arbitrary images:

- **RAW input:** a burst bundle containing RAW frames, camera characteristics, and motion metadata.
- **RGB input:** a bundle containing `rgb_volume` with shape `[T, C, H, W]`, `intrinsics`, and `rotations`. Enable this path with `--rgb_data`.

See `BundleDataset` in [train.py](train.py) for the exact bundle fields.

## Training with the Current Implementation

Run the launchers from the repository root. Each command optimizes one scene.

### Reflection Removal

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reflection.sh \
  --bundle_path data/path/to/frame_bundle.npz \
  --name reflection_example
```

### Occlusion Removal

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/occlusion.sh \
  --bundle_path data/path/to/frame_bundle.npz \
  --name occlusion_example
```

### Other Tasks

| Task | Launcher |
| :--- | :--- |
| Shadow removal | `scripts/shadow.sh` |
| Image dehazing | `scripts/dehaze.sh` |
| Video motion segmentation | `scripts/segmentation.sh` |
| Image fusion, an additional existing utility | `scripts/fusion.sh` |

Use the same `--bundle_path` and `--name` arguments with these launchers. For preprocessed RGB bundles, append `--rgb_data`:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/reflection.sh \
  --bundle_path data/path/to/rgb_bundle.npz \
  --name reflection_rgb_example \
  --rgb_data
```

For all available arguments, after installing the dependencies, run:

```bash
python train.py --help
```

The existing launchers select task-specific model settings and regularization weights. They have not yet been replaced with configurations for the manuscript's frequency-aware objective.

### Training Settings

The current defaults use Adam, a target batch of `2^18` sampled points, 80 batches per epoch, and 75 epochs. Individual launchers override settings such as the learning rate and grid sizes.

The manuscript reports approximately **4 minutes to fit a 42-frame sequence on one NVIDIA RTX 4090**. This is a reported experimental measurement, not a runtime guarantee for the current public code or other hardware.

## Outputs and Evaluation

The default final export renders the first reference frame. For an experiment named `reflection_example`, reconstructions are saved under:

```text
outputs/reflection_example-final/
├── reference.png          # Input observation at the reference time
├── transmission.png       # Reconstructed base layer
├── obstruction.png        # Reconstructed overlay layer
├── alpha.png              # Overlay alpha matte
├── combined.png           # Alpha-composited reconstruction
├── flow_transmission.npy  # Learned base-layer deformation
├── flow_obstruction.npy   # Learned overlay-layer deformation
└── metrics.txt            # Current observation-reconstruction metrics
```

The checkpoint and cached scene metadata are saved to:

```text
checkpoints/reflection_example/last.ckpt
checkpoints/reflection_example/bundle.pkl
```

**Evaluation scope:** The current training script and `test*.py` scripts compare `combined.png` with `reference.png`. This measures reconstruction of the observed mixture. Evaluating layer separation requires comparing the recovered base and overlay against their respective ground-truth layers, with a defined color-processing, alignment, and aggregation protocol. The existing metric logs should therefore not be treated as a reproduction of the manuscript's tables. The existing metric logs are legacy diagnostics. Their PSNR calculation currently subtracts and squares unsigned-integer image values and needs floating-point conversion before it can be used for quantitative reporting.

The interactive notebook is available at [tutorial.ipynb](tutorial.ipynb).

## Acknowledgments

This project builds on [Neural Spline Fields (NSF)](https://github.com/princeton-computational-imaging/NSF) and uses its released burst datasets. We thank the NSF authors for their code and data, and the [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) authors for the neural-network and hash-grid implementation.

