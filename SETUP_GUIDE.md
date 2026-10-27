# TreeAIBox Standalone Setup & Quickstart

This repository is configured on the **`TreeAIBox-Standalone`** branch with full **NVIDIA CUDA acceleration** on your NVIDIA GeForce RTX 5090 Laptop GPU.

---

## Environment Overview

- **Repository**: `arunbiju969/TreeAIBox`
- **Branch**: `TreeAIBox-Standalone`
- **Location**: `E:\Projects\TreeAIBox`
- **Python Version**: CPython 3.12 (managed in `.venv`)
- **PyTorch**: `2.11.0+cu128` (with CUDA 12.8 support)
- **GPU Detected**: `NVIDIA GeForce RTX 5090 Laptop GPU` (24 GB VRAM)
- **Key Installed Packages**:
  - `torch`, `torchvision`, `torchaudio` (CUDA 12.8 enabled)
  - `PyQt6`, `PyQt6-WebEngine`, `PySide6`
  - `pyvista`, `pyvistaqt`, `vtk`
  - `laspy`, `geopandas`, `pyogrio`, `shapely`, `pyshp`, `rasterio`
  - `scikit-learn`, `scikit-image`, `scipy`, `numpy`, `pandas`, `matplotlib`
  - `cut_pursuit_py`, `circle_fit`, `numpy_indexed`, `numpy_groupies`
  - `timm`, `litellm`, `requests`, `django`

---

## Launching Applications

All launchers live in the **`scripts\launch\`** folder (double-click the `.bat` files or run the `.ps1` scripts from anywhere in the repo).

### 1. Main Standalone Application (TreeAIBox GUI)
Run the batch file or PowerShell script:
- **Batch**: Double-click or run `scripts\launch\run_treeaibox.bat`
- **PowerShell**:
  ```powershell
  .\scripts\launch\run_treeaibox.ps1
  ```
- Or run directly:
  ```powershell
  .\.venv\Scripts\python.exe TreeAIBox.py
  ```

> In standalone mode, TreeAIBox operates independently without requiring CloudCompare. When started, it checks CUDA availability, detects your RTX 5090 Laptop GPU, and automatically checks the **"Use GPU (CUDA)"** option in the user interface.

### 2. Standalone Tree Visualizer GUI
Visualize individual tree point clouds and scalar fields (`itc`):
- **Modern variant (recommended)**:
  ```powershell
  .\scripts\launch\run_modern_tree_visu.ps1
  ```
- **Basic variant**:
  ```powershell
  .\.venv\Scripts\python.exe scripts\visualization\tree_visualizer_gui.py
  ```

### 3. Forestry AI Suite GUI (`treeAIbox_Enhancer.py` / `forestry_ai_suite.py`)
Multi-model tree visualization and analytics supporting **TreeAIBox**, **TreeLearn**, and **ForestFormer**. Features top category navigation ribbon, collapsible tree/layer panel, multi-layer LAS inspection, 2D polygon selection, trunk & branch reconstruction, volume estimation, GDB shapefile overlay, and AI quality evaluation:
- **Batch**: Double-click or run `scripts\launch\run_forestry_ai_suite.bat`
- **PowerShell**:
  ```powershell
  .\scripts\launch\run_forestry_ai_suite.ps1
  ```
- Or run directly:
  ```powershell
  .\.venv\Scripts\python.exe scripts\visualization\treeAIbox_Enhancer.py
  ```

### 4. Automated Segmentation Refiner (`refine_segmented_plots.py`)
Automated post-processor that cleans false stumps (< 2.5m, < 120 pts), disentangles overlapping trees with multiple stem cores, and updates inventory metrics:
- **Batch**: Double-click or run `scripts\launch\run_refine_plots.bat`
- **PowerShell**:
  ```powershell
  .\scripts\launch\run_refine_plots.ps1
  ```
- Or run directly:
  ```powershell
  .\.venv\Scripts\python.exe scripts\pointcloud_processing\refine_segmented_plots.py
  ```


---

## Verifying CUDA Acceleration

You can verify CUDA status anytime by running:
```powershell
.\.venv\Scripts\python.exe -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

Expected output:
```text
CUDA Available: True
Device: NVIDIA GeForce RTX 5090 Laptop GPU
```

---

## Model Zoo & Pretrained Weights

Pretrained model weights are stored in:
- Local Cache: `%LOCALAPPDATA%\CloudCompare\TreeAIBox\models`
- Repository: `models/`

The pretrained UAV and TreeIsoNet models (`treefiltering_uav`, `treeisonet_uav_mixedwood_crownoff`, `treeisonet_uav_mixedwood_stemcls`, `treeisonet_uav_mixedwood_treeloc`) have already been synced to the local model cache and verified for CUDA tensor inference. Additional models are automatically downloaded on demand when selected in the UI.
