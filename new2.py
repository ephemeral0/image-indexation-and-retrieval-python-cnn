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
            f"Rank {rank} [{mark}] {match_labels
