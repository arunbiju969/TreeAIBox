# TreeAIBox Scripts

This directory contains modular utility scripts, visualization GUIs, processing pipelines, and data tools for the **TreeAIBox** ecosystem.

---

## Directory Organization

```
scripts/
├── launch/                        # Double-click launchers (bat + ps1) for the main GUIs
│   ├── run_treeaibox.bat / .ps1              # TreeAIBox Standalone
│   ├── run_forestry_ai_suite.bat / .ps1      # Forestry AI Suite
│   ├── run_refine_plots.bat / .ps1           # Segmentation Refiner
│   └── run_modern_tree_visu.ps1              # Modern Tree Visualizer
│
├── visualization/                 # 3D interactive viewers & slice visualization tools
│   ├── treeAIbox_Enhancer.py     # Forestry AI Suite entry point (thin launcher)
│   ├── forestry_suite/           # Forestry AI Suite package (modularized):
│   │   ├── dialogs/              #   polygon, ITC assignment, batch params, GDB import, LAS compare, slice viewer
│   │   ├── window/               #   TreeVisualizerGUI assembled from 13 cohesive mixins
│   │   ├── analysis/             #   pure math engines (components, volume formulas, trunk trajectories)
│   │   ├── widgets.py            #   shared widgets (ModernButton)
│   │   ├── error_handling.py     #   lazy error logging + Qt exception capture
│   │   └── app.py                #   application entry point
│   ├── tree_visu.py              # Modern Tree Visualizer GUI (launch/run_modern_tree_visu.ps1)
│   ├── tree_visualizer_gui.py    # Basic Tree Visualizer GUI
│   ├── plane_cut_visualization.py# Interactive horizontal plane cut with RANSAC fitting
│   ├── simple_plane_visualization.py # Basic plane slice viewer
│   └── branch_split_detection.py # 8m height plane DBSCAN clustering
│
├── stem_reconstruction/           # Tree component extraction & circle/cylinder fitting
│   ├── component_extraction_connectivity.py # Advanced connectivity-based trunk & branch extraction (v2)
│   ├── component_extraction.py   # Baseline trunk and branch extraction (v1)
│   ├── multi_height_circle_fitting.py       # Multi-height stem cross-section circle fitting
│   ├── circle_fitting_reconstruction.py     # Taubin / Pratt / Kåsa / RANSAC circle fitting
│   └── generate_sample_stem.py   # Synthetic stem point cloud generator
│
├── field_assignment/              # Ground truth matching & GIS integration
│   ├── semantic_field_assignment.py         # Hungarian matching engine (spatial + DBH + height)
│   ├── semantic_field_assignment_gui.py     # PySide6/PyQt6 graphical matching interface
│   └── run_web_app.py            # Launcher for Django TreeToolBox Web Studio
│
├── volume_metrics/                # Post-processing volume statistics & center calculations
│   ├── volume_summary_report_gui.py         # PyQt6 volume JSON report generator & DBH calculator
│   └── extract_tree_centers.py   # CLI tool to extract XY centers & DBH to CSV / GeoJSON
│
├── pointcloud_processing/         # Point cloud filtering, cleaning, RGB repair & vectorization
│   ├── las_utils.py              # Unified tree ID detection, ITC dimension mapping & tree filtering
│   ├── sync_treeid_to_itc.py     # Batch CLI to sync 'treeID' to 'itc' scalar field
│   ├── refine_segmented_plots.py # Automated stem disentanglement & false tree removal
│   ├── las_to_polygon_gui.py     # PyQt6 GUI for generating 2D boundary polygons (Shapefile/GeoJSON)
│   ├── las_rgb_checker.py        # Inspects LAS/LAZ files for valid RGB channels
│   ├── rgb_restoration_tool.py   # Restores RGB channels from raw LAS to processed LAS via KDTree
│   ├── filter_by_scalar_field.py # Filters LAS points by scalar field criteria
│   ├── remove_ground_points.py   # Dedicated ground point removal filter
│   ├── apply_scalar_field_rules.py # Rule-based scalar field modifier (JSON/YAML)
│   ├── inspect_las.py            # Quick CLI LAS header and scalar field inspector
│   └── inspect_scalar_fields.py  # GUI/CLI scalar field distribution analyzer
│
├── Processing_WorkFlow/           # USDA Forest Service FUSION LiDAR processing pipeline
│   ├── run_fusion_workflow_gui.py# PySide6 FUSION workflow GUI
│   ├── run_fusion_workflow.py    # FUSION workflow engine
│   ├── groundfilter_gui.py       # GroundFilter & DTM generation GUI
│   ├── clip_plots_gui.py         # PolyClipData GUI
│   ├── gridmetrics_gui.py        # GridMetrics GUI
│   ├── cloudmetrics_gui.py       # CloudMetrics GUI
│   ├── normalize_las_gui.py      # LAS elevation normalization GUI
│   ├── app_settings.db           # SQLite settings store for workflow GUIs
│   └── workflow_guide.md         # Guide for extending FUSION workflows
│
├── Batch Files/                   # Windows batch scripts for FUSION CLI tools
│   ├── clip_plots.bat
│   ├── dtm2tif_sample.bat
│   ├── generate_dtm.bat
│   ├── groundfilter.bat
│   ├── normalize_stands.bat
│   ├── process_single_stand_grid.bat
│   ├── run_cloudmetrics.bat
│   └── run_gridmetrics_on_all_stands.bat
│
└── utils/                         # Maintenance and diagnostic tools
    └── find_empty_files.py       # Workspace empty file scanner
```

> Runtime exception logs from the Forestry AI Suite are written to `scripts/error_logs/`
> (gitignored, created only when an error occurs).

---

## Quick Launch Reference

### Main Applications (launchers in scripts\launch\)
```powershell
# TreeAIBox Standalone:
.\scripts\launch\run_treeaibox.ps1

# Forestry AI Suite (TreeAIBox / TreeLearn / ForestFormer):
.\scripts\launch\run_forestry_ai_suite.ps1

# Automated Segmentation Refiner:
.\scripts\launch\run_refine_plots.ps1
```

### Tree Visualizers
```powershell
# Modern Tree Visualizer GUI:
.\scripts\launch\run_modern_tree_visu.ps1

# Basic Tree Visualizer GUI:
.\.venv\Scripts\python.exe scripts\visualization\tree_visualizer_gui.py

# Forestry AI Suite (direct):
.\.venv\Scripts\python.exe scripts\visualization\treeAIbox_Enhancer.py
```

### Field Assignment
```powershell
# GUI Application:
.\.venv\Scripts\python.exe scripts\field_assignment\semantic_field_assignment_gui.py

# CLI Hungarian Matcher:
.\.venv\Scripts\python.exe scripts\field_assignment\semantic_field_assignment.py --help

# Django Web Studio:
.\.venv\Scripts\python.exe scripts\field_assignment\run_web_app.py
```

### Volume Summary Report
```powershell
.\.venv\Scripts\python.exe scripts\volume_metrics\volume_summary_report_gui.py
```

### Point Cloud to Polygon GUI
```powershell
.\.venv\Scripts\python.exe scripts\pointcloud_processing\las_to_polygon_gui.py
```
