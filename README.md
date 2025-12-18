# CryoSPARC TT-SVD / Tucker Denoising Pipeline

This directory contains a **CryoSPARC-integrated pipeline** for benchmarking
**tensor-based denoising methods (TT-SVD and Tucker)** on cryo-EM particle stacks.
The pipeline is designed to run **programmatically via the CryoSPARC Python API**
and to clean up intermediate jobs automatically.

---

## Overview

The workflow is:

1. Fetch cryo-EM datasets (optionally from EMPIAR)
2. Import particles into CryoSPARC
3. Perform 2D classification
4. Apply **TT-SVD or Tucker decomposition–based denoising** on particle stacks
   - Either on the full stack or per-class
5. Continue with:
   - Ab initio reconstruction
   - Homogeneous refinement
   - Local resolution estimation
6. Automatically delete intermediate CryoSPARC jobs to keep the workspace clean

The main orchestration logic lives in `pipeline.py`.

---

## Structure

```
.
├── pipeline.py       # Main pipeline entry point
├── cli_parse.py      # Command-line argument parsing
├── ttsvd_denoise.py  # TT-SVD / Tucker stack denoising implementation
├── cs_wrappers.py    # Thin wrappers around CryoSPARC Python API jobs
├── utils.py          # Utility functions (box size heuristics, helpers)
├── fetch_EMPIAR.sh   # Script to download EMPIAR datasets
├── run_benchmarks.sh # Script to benchmark pipeline on datasets
└── README.md
```

---

## Requirements

- Python 3.9+
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [mrcfile](https://mrcfile.readthedocs.io/)
- [tntorch](https://tntorch.readthedocs.io/)
- [torch](https://pytorch.org/)
- [CryoSPARC API and tools](https://guide.cryosparc.com/setup/cryosparc-setup) – for creating and managing CryoSPARC jobs in Python.


The code assumes it is executed **inside a CryoSPARC-enabled environment**
with valid project and workspace access.

---

## Main Pipeline (`pipeline.py`)

`pipeline.py` is the **entry point**.

It performs:
- Project/workspace lookup
- Particle import
- 2D classification
- Stack denoising using TT-SVD or Tucker decomposition
- Downstream CryoSPARC jobs
- Cleanup of intermediate jobs

The pipeline **tracks all job UIDs** and deletes them at the end.

### Usage Example

```bash
python pipeline.py \
    --project P4 \
    --workspace W1 \
    --tsv empiar_dataset.tsv \
    --dataset_dir MRC_data \
    --hostname cryosparc.local \
    --mrc_ttsvd_job --mrc_ranks 24 \
    --cls_ttsvd_job --cls_ranks 12
```

---

## Denoising Methods (`ttsvd_denoise.py`)

This module implements **tensor-based denoising** of CryoSPARC micrographs and particle stacks using:

- **TT-SVD** (`tt`);
- **Tucker** decomposition (`tucker`).

Denoising can be applied to:

- Individual **micrographs** (via `process_file`);
- Full **particle stacks** or **2D-class stacks** (via `process_stack`);

Tensor ranks are controlled via function arguments or CLI parameters.

### Key Functions

#### Micrograph Denoising
- `process_file(path, outdir, patch, stride, ranks, eps, device, decomposition="tt")` - Applies patch-wise TT-SVD/Tucker
 denoising to a **single 2D micrograph** (.mrc file) and saves the denoised output in `outdir`;
- `denoise_image(img, patch=256, stride=128, ranks=24, eps=None, device='cpu', decomposition="tt")` - Core patch-based
 denoising of a 2D NumPy array. Internal function used by `process_file`;

#### Particle / Class Stack Denoising

- `process_stack(stack, patch=64, stride=32, ranks=None, eps=None, device='cpu', decomposition="tt")` - Denoises a 3D
 particle stack (`num_particles × H × W`) in a patch-wise fashion and returns a denoised NumPy array. Used for
 per-class stacks or full particle stacks;

#### Tensor Utilities

- `decompose_tensor(tensor, decomposition="tt", ranks=None, eps=None, device="cpu")` - Performs the actual TT-SVD
 or Tucker decomposition on a 2D or 3D tensor;
- `compute_tt_patch(patch, ranks=None, eps=None, device='cpu', decomposition="tt")` - Decomposes and reconstructs
 a single patch (internal function);
- `patch_grid(H, W, patch, stride)` - Generates top-left coordinates for sliding patches over an image;
- `hann2d(h, w)` - Creates a 2D Hann window for smooth blending of overlapping patches;
- `factors(n)` - Helper to factorize image dimensions for patch reshaping;
- `load_mrc(path)` / `save_mrc(path, arr)` - Read/write `.mrc` files as float32 NumPy arrays.

### Notes

- Micrograph images and particle stacks are expected as float32 NumPy arrays internally; `.mrc` files are converted automatically.
- Patch size and stride control memory usage and blending smoothness:
  - Default for micrographs: patch=256, stride=128
  - Default for stacks: patch=64, stride=32
- GPU acceleration is supported via `device="cuda"`.

---

## Command-Line Interface (`cli_parse.py`)

This module handles CLI parsing for the CryoSPARC processing pipeline.
It provides argument parsing for specifying datasets,
CryoSPARC projects/workspaces, and tensor decomposition options.

### Key Arguments

**Project / Dataset**
- `--project` (required): CryoSPARC project ID (e.g., `P4`);
- `--workspace` (optional): CryoSPARC workspace ID (e.g., `W1`);
- `--tsv` (required): Path to EMPIAR TSV file describing micrographs/particles;
- `--dataset_dir` (required): Directory containing MRC files;
- `--hostname` (required): Hostname where CryoSPARC ports are open;

**MRC Decomposition**
- `--mrc_ttsvd_job` / `--mrc_tucker_job`: Mutually exclusive flags to choose decomposition method for micrographs;
- `--mrc_ranks`: Optional tensor ranks for MRC decomposition;

**Class Stack (CLS) Decomposition**
- `--cls_ttsvd_job` / `--cls_tucker_job`: Mutually exclusive flags to choose decomposition method for 2D class stacks;
- `--cls_ranks`: Optional tensor ranks for CLS decomposition;

### Helper Functions

- `parse_cli()`: Returns parsed CLI arguments;
- `get_decomposition(args, stage)`: Determines decomposition type (`tt` or `tucker`) and ranks based on CLI flags for a given stage (`mrc` or `cls`);

---

## CryoSPARC Integration (`cs_wrappers.py`)

This module provides high-level wrappers for creating, queuing, and managing CryoSPARC jobs programmatically. 
Each function handles job submission, waits for completion, and returns the job UID.

### Available Wrappers

**Import and Preprocessing**
- `run_import_files(entry, kind, ws)` — Import movies or micrographs from EMPIAR datasets;
- `run_motion_correction(job_uid, ws)` — Patch-based motion correction;
- `run_mrc_TTSVD(project, job_uid, ws, decomp='tt', ranks=24)` — TT-SVD/Tucker denoising of micrographs;
- `run_ctf_est(job_uid, project, ws)` — CTF estimation;
- `run_ppicking(entry, job_uid, ws)` — Blob-based particle picking;
- `run_pextract(entry, job_uid, ws)` — Extract particles into boxed stacks;

**2D Classification**
- `run_2DClass(job_uid, ws, filam_flag=False)` — Standard 2D classificatio;.
- `run_2Dmanual(job_uid, ws)` — Manual selection of 2D classes;
- `run_stack_TTSVD(project, job_uid, ws, decomp='tt', ranks=24)` — TT-SVD/Tucker denoising of 2D class stacks;

**3D Reconstruction**
- `run_abinitio(job_uid, project, ws)` — Ab initio reconstruction;
- `run_homoref(job_uid, ws)` — Homogeneous refinement;
- `run_lrdist_estim(job_uid, ws)` — Local resolution estimation.

---

## Utility Logic (`utils.py`)

This module contains internal utility functions for CryoSPARC data handling. 

### Key functionalities

- **Job management**
  - `delete_job(job_uid, project_uid)` — Delete a CryoSPARC job programmatically;

- **Micrograph & particle helpers**
  - `_get_micro_params(entry)` — Extract EMPIAR ID, pixel size, voltage, spherical aberration, and total exposure from metadata;
  - `_get_particle_diams(entry)` — Returns min/max particle diameters;
  - `_get_picker_params(entry, margin=0.3, appx=16)` — Computes box sizes and crop sizes for particle picking;

- **Class-to-particle mapping**
  - `_get_rows_per_class(dset)` — Groups dataset rows by 2D class ID;

- **Stack and patch utilities**
  - `_crop_with_padding(img, cx, cy, box)` — Crop a patch with zero-padding if needed;
  - `_get_particle_patch(row, micrographs)` — Extract a particle patch from a micrograph/stack;
  - `_paste_patch(img, patch, cx, cy, box, idx=None)` — Paste a patch back into an image/stack.

---

## EMPIAR Download Script (`fetch_EMPIAR.sh`)

Downloads TIFF/MRC micrographs from EMPIAR via FTP.

Usage:
```bash
./fetch_EMPIAR.sh <EMPIAR_ID>
```

---

## Benchmarking Script (`run_benchmarks.sh`)

Runs a grid benchmark over denoising modes (no denoising / TT-SVD / Tucker),
applied at two stages:

- MRC stack level
- Per-2D-class stack level

For each combination, the script launches a CryoSPARC pipeline via `pipeline.py`. Usage:
```bash
./run_benchmarks.sh <DATASET> <PROJECT_UID> <PROJECT_WD> <PROJECT_NAME> <WORKSPACE> <META_TSV> <RANKS>
```

Arguments:

- `DATASET` - EMPIAR dataset identifier or local dataset directory name. Used both for 
downloading data and as the working directory;
- `PROJECT_UID` - CryoSPARC project UID (e.g. P12);
- `PROJECT_WD` - CryoSPARC project working directory on disk;
- `PROJECT_NAME` - CryoSPARC project name (used for cleanup);
- `WORKSPACE` - CryoSPARC workspace UID;
- `META_TSV` - a TSV file describing datset metadata (passed to the pipeline),
an example is included in `cryoEM_data.tsv`;
- `RANKS` - tensor ranks used for both TT-SVD and Tucker decompositions.

Notes:
- Assumes CryoSPARC project/workspace are already configured
- Removes previous working directories before running
- Intended for controlled benchmarking, not production use

---

## Important Notes

- Intermediate CryoSPARC jobs are deleted automatically
- The pipeline assumes single-user control of the target workspace
- This code is intended for method development and benchmarking,
not turn-key production processing
