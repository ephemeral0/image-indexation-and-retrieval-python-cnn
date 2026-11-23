import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import torchvision.models as models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd

pretreatment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

#weights = models.ResNet18_Weights.DEFAULT
#model = models.resnet18(weights=weights)
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1]) 
model.eval() 


def extraction(image):
    image = pretreatment(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        features = model(image)

    return features.flatten().numpy()    

def dataset_treatment(directory):
    features = []
    img_path = []

    for file in directory.iterdir():
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image = Image.open(file).convert("RGB")

            vector = extraction(image)
            features.append(vector)
            img_path.append(str(file))

    return np.array(features), img_path 

def load_or_extract_features(directory):

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if FEATURES_FILE.exists() and PATHS_FILE.exists():

        print("Loading saved features...")

        features = np.load(FEATURES_FILE)
        paths = np.load(PATHS_FILE)

    else:

        print("Extracting features for the first time...")

        features, paths = dataset_treatment(directory)

        np.save(FEATURES_FILE, features)
        np.save(PATHS_FILE, paths)

        print("Features extracted and saved!")

    print("Features shape:", features.shape)
    print("Number of images:", len(paths))

    return features, paths

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"

# Dataset directories
COREL_DIR = DATASETS_DIR / "Corel-1K"
FASHION_DIR = DATASETS_DIR / "FASHION-MNIST"
CIFAR_DIR = DATASETS_DIR / "CIFAR-100"
TESTING_DIR = DATASETS_DIR / "TESTING"

# Features directory
FEATURES_DIR = BASE_DIR / "features"

# Saved ResNet-50 features
FEATURES_FILE = FEATURES_DIR / "corel_resnet50_features.npy"
PATHS_FILE = FEATURES_DIR / "corel_resnet50_paths.npy"


def inspect_dataset(name, directory):
    print(f"\n{'=' * 40}")
    print(f"DATASET: {name}")
    print(f"{'=' * 40}")

    if not directory.exists():
        print("Directory not found!")
        return

    files = list(directory.rglob("*"))

    files = [f for f in files if f.is_file()]

    print(f"Total files: {len(files)}")

    extensions = {}

    for file in files:
        ext = file.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1

    print("File types:")

    for ext, count in extensions.items():
        print(f"{ext or '[no extension]'}: {count}")


def search_similar_images(query_path, features, paths, top_k):

    # Extract features from the query image
    query_image = Image.open(query_path).convert("RGB")
    query_features = extraction(query_image)

    # Calculate cosine similarity
    scores = cosine_similarity(
        query_features.reshape(1, -1),
        features
    )[0]

    # Exclude the query itself if it belongs to the dataset
    query_resolved = Path(query_path).resolve()

    valid_indices = [
        i for i, path in enumerate(paths)
        if Path(path).resolve() != query_resolved
    ]

    # Sort by similarity, highest first
    ranked_indices = sorted(
        valid_indices,
        key=lambda i: scores[i],
        reverse=True
    )

    # Keep only the Top-K results
    top_indices = ranked_indices[:top_k]

    return query_image, top_indices, scores


def display_results(query_image, top_indices, scores, paths):

    fig, axes = plt.subplots(3, 4, figsize=(10, 12))

    # Flatten axes into a 1D array 
    axes = axes.flatten()

    # Display query image
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    # Display retrieved images
    for position, index in enumerate(top_indices ):

        image = Image.open(paths[index]).convert("RGB")

        axes[position + 1].imshow(image)
        axes[position + 1].set_title(
            f"Similarity: {scores[index]:.2%}"
        )
        axes[position + 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    inspect_dataset("Corel-1K", COREL_DIR)

    # Load saved features or extract them if they don't exist
    features, paths = load_or_extract_features(COREL_DIR)

    # Select query image
    query_path = COREL_DIR / "166.jpg"

    # Search for similar images
    query_image, top_indices, scores = search_similar_images(
        query_path,
        features,
        paths,
        top_k=11
    )

    # Display results
    display_results(query_image, top_indices, scores, paths)