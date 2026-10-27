#!/usr/bin/env python3
"""
Sync / add 'itc' scalar field from 'treeID' in LAS/LAZ files.

This utility ensures compatibility with legacy tools that strictly require
an 'itc' dimension by copying tree IDs from 'treeID' (or vice versa).
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import laspy
import numpy as np

# Ensure root workspace is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from scripts.pointcloud_processing.las_utils import ensure_itc_dimension, find_tree_id_dimension, get_unique_tree_ids


def add_itc_field_to_file(file_path, output_path=None, in_place=True):
    """Ensure an 'itc' dimension exists in the LAS/LAZ file, copied from treeID."""
    if output_path is None and not in_place:
        raise ValueError("Must specify output_path when in_place=False")

    save_path = file_path if in_place else output_path

    try:
        las = laspy.read(file_path)
        dim_names = list(las.point_format.dimension_names)

        if 'itc' in dim_names:
            print(f"[SKIP] '{os.path.basename(file_path)}' already contains 'itc' dimension.")
            return True

        success, source_field = ensure_itc_dimension(las, target_dim='itc')
        if not success or not source_field:
            print(f"[WARN] No tree ID field found in '{os.path.basename(file_path)}'. Skipping.")
            return False

        las.write(save_path)
        unique_trees = get_unique_tree_ids(np.array(las.itc))
        print(f"[OK] Added 'itc' field to '{os.path.basename(save_path)}' ({len(unique_trees)} trees mapped from '{source_field}').")
        return True

    except Exception as e:
        print(f"[ERROR] Failed processing '{file_path}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Add 'itc' scalar field to LAS/LAZ files based on 'treeID'.")
    parser.add_argument(
        "--path",
        "-p",
        default=r"E:\Data\TreeLearn_Results\full_forest",
        help="Path to folder or single LAS/LAZ file"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before modifying files in-place"
    )

    args = parser.parse_args()

    target = args.path
    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.las")) + glob.glob(os.path.join(target, "*.laz")))
    else:
        print(f"Error: Target path not found: {target}")
        sys.exit(1)

    print(f"Processing {len(files)} file(s) in: {target}")
    success_count = 0
    for f in files:
        if add_itc_field_to_file(f, in_place=True):
            success_count += 1

    print(f"\nDone! Successfully updated {success_count}/{len(files)} files.")


if __name__ == "__main__":
    main()
