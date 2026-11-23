import os
import sys
import subprocess
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
COREL_ROOT = "/home/leticia/Documents/Uni/M2/IRDM/corel/"

DB_FILES = {
    "resnet18": "/home/leticia/Documents/Uni/M2/IRDM/project/corel_features_resnet18.npz",
    "squeezenet1_1": "/home/leticia/Documents/Uni/M2/IRDM/project/corel_features_squeezenet1_1.npz",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def select_query_image(initial_dir=COREL_ROOT):
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
        return None

    return result.stdout.strip()


def load_feature_extractor(model_name):
    """Loads either 'resnet18' or 'squeezenet1_1' to extract the query image's vector."""
    if model_name == "resnet18":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.fc = torch.nn.Identity()
    elif model_name == "squeezenet1_1":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(DEVICE)
    model.eval()
    return model


def extract_single_vector(image_path, model, model_name):
    """Preprocesses the query image and returns an L2-normalized 512D vector."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if model_name == "resnet18":
            features = model(tensor)
        elif model_name == "squeezenet1_1":
            conv_maps = model.features(tensor)
            features = torch.nn.functional.adaptive_avg_pool2d(conv_maps, (1, 1))

        vec = features.squeeze().cpu().numpy()

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

    return vec


def visualize_results(query_path, top_indices, db_paths, db_classes, distances, model_name):
    """Displays the Query Image and Top-10 retrieved matches in a 3x4 grid."""
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    axes = axes.flatten()

    query_class = os.path.basename(os.path.dirname(query_path))
    axes[0].imshow(Image.open(query_path).convert("RGB"))
    axes[0].set_title(
        f"QUERY ({query_class})\n{os.path.basename(query_path)}",
        fontsize=11,
        fontweight="bold",
        color="crimson",
    )
    axes[0].axis("off")

    for rank, idx in enumerate(top_indices, start=1):
        matched_path = db_paths[idx]
        matched_class = db_classes[idx]
        ax = axes[rank]
        ax.imshow(Image.open(matched_path).convert("RGB"))

        is_match = "✓" if matched_class == query_class else "✗"
        ax.set_title(
            f"Rank {rank} [{is_match}] ({matched_class})\n{os.path.basename(matched_path)} | Dist: {distances[idx]:.4f}",
            fontsize=9,
        )
        ax.axis("off")

    axes[11].axis("off")
    fig.suptitle(f"Top-10 Similar Images — {model_name.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 3. INTERACTIVE RETRIEVAL MENU (READ-ONLY FROM SAVED .NPZ BASES)
# ==============================================================================
def run_interactive_menu():
    loaded_cache = {}  # Stores loaded models and .npz arrays in RAM for fast repeat queries

    while True:
        print("\n==========================================")
        print("      COREL IMAGE RETRIEVAL SYSTEM        ")
        print("==========================================")
        print("  1. ResNet-18      (D = 512)")
        print("  2. SqueezeNet 1.1 (D = 512)")
        print("  3. Quit")
        print("==========================================")

        choice = input("Select a network [1-3]: ").strip()

        if choice == "3":
            print("Exiting program. Goodbye!")
            break
        elif choice == "1":
            model_name = "resnet18"
        elif choice == "2":
            model_name = "squeezenet1_1"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

        db_file = DB_FILES[model_name]
        if not os.path.exists(db_file):
            print(f"[ERROR] Saved feature database '{db_file}' not found in current directory.")
            continue

        # Load pre-computed .npz database and query extractor once per network
        if model_name not in loaded_cache:
            print(f"Reading pre-computed features from '{db_file}'...")
            with np.load(db_file) as data:
                db_matrix = data["features"]
                db_paths = data["paths"]
                db_classes = data["class_folders"]

            print(f"Loaded {len(db_paths)} vectors (shape: {db_matrix.shape}). Loading {model_name.upper()} for query extraction...")
            model = load_feature_extractor(model_name)
            loaded_cache[model_name] = (model, db_matrix, db_paths, db_classes)

        model, db_matrix, db_paths, db_classes = loaded_cache[model_name]

        # Open KDE file dialog to pick a query image
        print("Opening file picker to select a query image...")
        query_path = select_query_image(initial_dir=COREL_ROOT)
        if query_path is None:
            print("No image selected. Returning to menu.")
            continue

        print(f"Selected Query: {query_path}")

        # Extract query vector and compute Cosine distance against the loaded database
        vec_query = extract_single_vector(query_path, model, model_name).reshape(1, -1)
        distances = cdist(vec_query, db_matrix, metric="cosine")[0]

        # Exclude the query image itself if it exists inside the database
        abs_query_path = os.path.abspath(query_path)
        for idx, p in enumerate(db_paths):
            if os.path.abspath(p) == abs_query_path:
                distances[idx] = np.inf
                break

        # Rank and display Top-10 closest matches
        top_10_indices = np.argsort(distances)[:10]

        print(f"\n--- Top 10 Matches ({model_name.upper()}) ---")
        for rank, idx in enumerate(top_10_indices, start=1):
            print(
                f"Rank {rank:2d}: {os.path.basename(db_paths[idx])} "
                f"(Class: {db_classes[idx]}) | Cosine Dist: {distances[idx]:.5f}"
            )

        visualize_results(query_path, top_10_indices, db_paths, db_classes, distances, model_name)

        again = input("\nDo you want to search another image? [y/n]: ").strip().lower()
        if again not in ("y", "yes"):
            print("Exiting program. Goodbye!")
            break


if __name__ == "__main__":
    run_interactive_menu()
