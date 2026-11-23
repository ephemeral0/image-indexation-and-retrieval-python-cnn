import os
import sys
import tkinter as tk
from tkinter import filedialog
import subprocess
import numpy as np

#going through Corel's classes
def collect_corel_image_paths(root_dir, save_npz_path=None):
    """
    Recursively traverses Corel 1k, 5k, and 10k folders and their class subfolders
    to collect all image file paths and infer their class/subset metadata.
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_paths = []
    subsets = []       # e.g., 'Corel 1k', 'Corel 5k', 'Corel 10k'
    class_folders = [] # Name of the immediate parent class folder

    for current_root, dirs, files in os.walk(root_dir):
        # Sort directories and files in-place so traversal order is deterministic
        dirs.sort()
        files.sort()

        for fname in files:
            if fname.lower().endswith(valid_extensions):
                full_path = os.path.join(current_root, fname)
                image_paths.append(full_path)

                # Extract relative structure: <root_dir>/<subset>/<class_folder>/<image>
                rel_path = os.path.relpath(full_path, root_dir)
                parts = rel_path.split(os.sep)

                subset_name = parts[0] if len(parts) > 1 else "root"
                folder_name = os.path.basename(current_root)

                subsets.append(subset_name)
                class_folders.append(folder_name)

    if save_npz_path is not None:
        np.savez(
            save_npz_path,
            paths=np.array(image_paths),
            subsets=np.array(subsets),
            class_folders=np.array(class_folders),
        )
        print(f"Saved {len(image_paths)} image paths to '{save_npz_path}'.")

    return image_paths, subsets, class_folders



def select_query_image(initial_dir="/home/leticia/Documents/images"):
    """Opens KDE Plasma's native file picker (kdialog) with full thumbnail support."""
    start_path = initial_dir if os.path.exists(initial_dir) else os.path.expanduser("~")

    cmd = [
        "kdialog",
        "--getopenfilename",
        start_path,
        "Image Files (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)",
        "--title",
        "Select a Query Image",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not result.stdout.strip():
        print("No image selected. Exiting.")
        sys.exit(0)

    return result.stdout.strip()


COREL_ROOT = "/home/leticia/Documents/Uni/M2/IRDM/corel/Corel-1K"  # Folder containing Corel 1k, 5k, 10k
SAVE_FILE = "corel_all_paths.npz"

all_paths, all_subsets, all_classes = collect_corel_image_paths(COREL_ROOT, save_npz_path=SAVE_FILE)

print(f"\nTotal images collected: {len(all_paths)}")
print(f"\nSubsets found: {sorted(set(all_subsets))}")
print(f"\nUnique class folders found: {len(set(all_classes))}")
if all_paths:
    print(f"\nSample entry: {all_paths[0]} -> Subset: {all_subsets[0]}, Class: {all_classes[0]}")


# QUERY_IMG_PATH = select_query_image(initial_dir="/home/leticia/Documents/Uni/M2/IRDM")
# print(QUERY_IMG_PATH)
