import os
import sys
import subprocess
import numpy as np
import torch
from torchvision import transforms
from PIL import Image



COREL_ROOT = "/home/leticia/Documents/Uni/M2/IRDM/corel/"
PATHS_FILE = "corel_all_paths.npz"



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SUBSET = None


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#going through Corel's classes
def collect_corel_image_paths(root_dir, save_npz_path=None):
    print("\nCollecting paths...\n")
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_paths = []
    subsets = []       # e.g., 'Corel 1k', 'Corel 5k', 'Corel 10k'
    class_folders = [] # Name of the immediate parent class folder
    i=0
    for current_root, dirs, files in os.walk(root_dir):
        # Sort directories and files in-place so traversal order is deterministic
        dirs.sort()
        files.sort()

        for fname in files:
            if fname.lower().endswith(valid_extensions):
                full_path = os.path.join(current_root, fname)
                image_paths.append(full_path)
                i+=1
                if i%1000 == 0: #change len dirs
                    print(f"\nProcessed : {i}/16000\n")
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

def load_feature_extractor(model_name):
    """Loads either 'resnet18' or 'squeezenet1_1' configured for D=512 feature extraction."""
    if model_name == "resnet18":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.fc = torch.nn.Identity()  # Outputs (1, 512) directly[cite: 2]
    elif model_name == "squeezenet1_1":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
        # SqueezeNet uses model.features (1, 512, 13, 13) -> needs global average pooling[cite: 1]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(DEVICE)
    model.eval()
    return model

def extract_single_vector(image_path, model, model_name):
    """Preprocesses one image, runs forward pass, and returns an L2-normalized 512D vector."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if model_name == "resnet18":
            features = model(tensor)  # Shape: (1, 512)[cite: 2]
        elif model_name == "squeezenet1_1":
            conv_maps = model.features(tensor)  # Shape: (1, 512, 13, 13)[cite: 1]
            features = torch.nn.functional.adaptive_avg_pool2d(conv_maps, (1, 1))  # Shape: (1, 512, 1, 1)[cite: 1]

        vec = features.squeeze().cpu().numpy()

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

    return vec


def build_and_save_feature_db(model_name, paths, subsets, class_folders, output_npz):
    """Extracts features for all provided paths using `model_name` and saves to `output_npz`."""
    print(f"\n============================================================")
    print(f"Building Feature Database for: {model_name.upper()} (Device: {DEVICE})")
    print(f"============================================================")

    model = load_feature_extractor(model_name)
    feature_vectors = []
    valid_indices = []

    total = len(paths)
    for idx, img_path in enumerate(paths):
        try:
            vec = extract_single_vector(img_path, model, model_name)
            feature_vectors.append(vec)
            valid_indices.append(idx)
        except Exception as e:
            print(f"[WARNING] Skipping corrupted/unreadable image '{img_path}': {e}")

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"  Processed {idx + 1}/{total} images...")

    feature_matrix = np.array(feature_vectors, dtype=np.float32)

    np.savez(
        output_npz,
        features=feature_matrix,
        paths=paths[valid_indices],
        subsets=subsets[valid_indices],
        class_folders=class_folders[valid_indices],
        model_name=model_name,
    )
    print(f"Saved '{output_npz}' -> Matrix shape: {feature_matrix.shape}")


# all_paths, all_subsets, all_classes = collect_corel_image_paths(COREL_ROOT, save_npz_path=PATHS_FILE)
#
# print(f"\nTotal images collected: {len(all_paths)}")
# print(f"\nSubsets found: {sorted(set(all_subsets))}")
# print(f"\nUnique class folders found: {len(set(all_classes))}")
# if all_paths:
#     print(f"\nSample entry: {all_paths[0]} -> Subset: {all_subsets[0]}, Class: {all_classes[0]}")


# 1. Collect paths (or load existing paths file if already generated)
# if os.path.exists(PATHS_FILE):
#     print(f"Loading existing paths from '{PATHS_FILE}'...")
#     with np.load(PATHS_FILE) as data:
#         paths = data["paths"]
#         subsets = data["subsets"]
#         class_folders = data["class_folders"]
#else:
    #paths, subsets, class_folders = collect_corel_image_paths(COREL_ROOT, PATHS_FILE) # Optional subset filtering (e.g., only 'Corel 10k')
if TARGET_SUBSET is not None:
    mask = subsets == TARGET_SUBSET
    paths = paths[mask]
    subsets = subsets[mask]
    class_folders = class_folders[mask]
    print(f"Filtered to subset '{TARGET_SUBSET}': {len(paths)} images.")


QUERY_IMG_PATH = select_query_image(initial_dir="/home/leticia/Documents/Uni/M2/IRDM")
print(QUERY_IMG_PATH)
# vecQuery = extract_single_vector(QUERY_IMG_PATH,model,"squeezenet")
#distances = cdist(vecQuery, db_matrix, metric="cosine")[0]
#top_10_indices = np.argsort(distances)[:10]


#distance calculations

#ranking




# 2. Extraction and saving ResNet-18 database
# build_and_save_feature_db(
#     model_name="resnet18",
#     paths=paths,
#     subsets=subsets,
#     class_folders=class_folders,
#     output_npz="corel_features_resnet18.npz",
# )>

# 3. Extraction and saving SqueezeNet 1.1 database
# build_and_save_feature_db(
#     model_name="squeezenet1_1",
#     paths=paths,
#     subsets=subsets,
#     class_folders=class_folders,
#     output_npz="corel_features_squeezenet1_1.npz",
# )
