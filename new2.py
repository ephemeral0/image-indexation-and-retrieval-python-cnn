import os
import sys
import subprocess
import urllib.request
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

IMAGENET_LABELS_FILE = "/home/leticia/Documents/Uni/M2/IRDM/project/imagenet_classes.txt"

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
def load_imagenet_labels():
    """Downloads (once) and loads the 1,000 ImageNet class names."""
    if not os.path.exists(IMAGENET_LABELS_FILE):
        print("Downloading ImageNet class labels list...")
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, IMAGENET_LABELS_FILE)
    with open(IMAGENET_LABELS_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]


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


def load_dual_model(model_name):
    """Loads 'resnet18' or 'squeezenet1_1' keeping the final 1000-class classifier intact."""
    if model_name in ("resnet18", "squeezenet1_1"):
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.to(DEVICE)
    model.eval()
    return model


def extract_features_and_labels(image_path, model, model_name, imagenet_labels, top_k=3):
    """Returns (L2-normalized 512D feature vector, list of Top-K predicted ImageNet labels)."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if model_name == "resnet18":
            x = model.conv1(tensor)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            pooled = model.avgpool(x)                     # Shape: (1, 512, 1, 1)
            logits = model.fc(torch.flatten(pooled, 1))   # Shape: (1, 1000)

        elif model_name == "squeezenet1_1":
            conv_maps = model.features(tensor)            # Shape: (1, 512, 13, 13)
            pooled = torch.nn.functional.adaptive_avg_pool2d(conv_maps, (1, 1))
            logits = model.classifier(conv_maps).view(1, -1)  # Shape: (1, 1000)

        # 1. L2-normalized 512D descriptor for Cosine distance ranking
        vec = pooled.squeeze().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        # 2. Top-K predicted ImageNet class names from the final classification layer
        probs = torch.nn.functional.softmax(logits[0], dim=0)
        _, top_ids = torch.topk(probs, top_k)
        predicted_labels = [imagenet_labels[i] for i in top_ids.cpu().tolist()]

    return vec, predicted_labels


def visualize_results(query_path, query_labels, match_results, distances, model_name):
    """Displays the Query Image and Top-10 retrieved matches judged by deduced ImageNet labels."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].imshow(Image.open(query_path).convert("RGB"))
    axes[0].set_title(
        f"QUERY: {query_labels[0]}\n({os.path.basename(query_path)})",
        fontsize=11,
        fontweight="bold",
        color="crimson",
    )
    axes[0].axis("off")

    for rank, (idx, matched_path, match_labels, is_match) in enumerate(match_results, start=1):
        ax = axes[rank]
        ax.imshow(Image.open(matched_path).convert("RGB"))

        mark = "✓" if is_match else "✗"
        ax.set_title(
            f"Rank {rank} [{mark}] {match_labels[0]}\n{os.path.basename(matched_path)} | Dist: {distances[idx]:.4f}",
            fontsize=9,
        )
        ax.axis("off")

    axes[11].axis("off")
    fig.suptitle(
        f"Top-10 Similar Images — {model_name.upper()} (Query Top-3: {', '.join(query_labels)})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. INTERACTIVE RETRIEVAL MENU (READ-ONLY FROM SAVED .NPZ BASES)
# ==============================================================================
def run_interactive_menu():
    loaded_cache = {}  # Stores loaded models and .npz arrays in RAM for fast repeat queries
    imagenet_labels = load_imagenet_labels()

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
            print(f"[ERROR] Saved feature database '{db_file}' not found.")
            continue

        # Load pre-computed .npz database and full network once per architecture
        if model_name not in loaded_cache:
            print(f"Reading pre-computed features from '{db_file}'...")
            with np.load(db_file) as data:
                db_matrix = data["features"]
                db_paths = data["paths"]
                db_classes = data["class_folders"]

            print(f"Loaded {len(db_paths)} vectors (shape: {db_matrix.shape}). Loading {model_name.upper()}...")
            model = load_dual_model(model_name)
            loaded_cache[model_name] = (model, db_matrix, db_paths, db_classes)

        model, db_matrix, db_paths, db_classes = loaded_cache[model_name]

        # Open KDE file dialog to pick a query image
        print("Opening file picker to select a query image...")
        query_path = select_query_image(initial_dir=COREL_ROOT)
        if query_path is None:
            print("No image selected. Returning to menu.")
            continue

        print(f"Selected Query: {query_path}")

        # Extract query 512D vector AND deduced Top-3 ImageNet labels
        vec_query, query_labels = extract_features_and_labels(
            query_path, model, model_name, imagenet_labels, top_k=3
        )
        print(f"Query Deduced Labels (Top-3): {query_labels}")

        # Compute Cosine distance against the pre-saved database
        distances = cdist(vec_query.reshape(1, -1), db_matrix, metric="cosine")[0]

        # Exclude the query image itself if it exists inside the database
        abs_query_path = os.path.abspath(query_path)
        for idx, p in enumerate(db_paths):
            if os.path.abspath(p) == abs_query_path:
                distances[idx] = np.inf
                break

        # Rank Top-10 closest matches and deduce their semantic labels
        top_10_indices = np.argsort(distances)[:10]
        match_results = []
        true_matches = 0

        print(f"\n--- Top 10 Matches ({model_name.upper()}) ---")
        for rank, idx in enumerate(top_10_indices, start=1):
            matched_path = db_paths[idx]
            _, match_labels = extract_features_and_labels(
                matched_path, model, model_name, imagenet_labels, top_k=3
            )

            # Check if query and candidate share at least one Top-3 ImageNet label
            shared_labels = set(query_labels).intersection(set(match_labels))
            is_match = len(shared_labels) > 0
            if is_match:
                true_matches += 1

            mark = "✓" if is_match else "✗"
            match_results.append((idx, matched_path, match_labels, is_match))

            print(
                f"Rank {rank:2d} [{mark}]: {os.path.basename(matched_path):12s} | "
                f"Deduced: {match_labels[0]:20s} (Folder: {db_classes[idx]}) | Dist: {distances[idx]:.5f}"
            )

        print(f"--> Semantic Matches (Top-3 Label Overlap): {true_matches}/10 (Precision@10 = {true_matches / 10:.2f})")

        visualize_results(query_path, query_labels, match_results, distances, model_name)

        again = input("\nDo you want to search another image? [y/n]: ").strip().lower()
        if again not in ("y", "yes"):
            print("Exiting program. Goodbye!")
            break


if __name__ == "__main__":
    run_interactive_menu()
