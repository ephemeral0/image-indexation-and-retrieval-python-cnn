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

def dataset_treatment(csv_file, limit=None):

    df = pd.read_csv(csv_file)

    # Limit the number of images for testing
    if limit is not None:
        df = df.iloc[:limit]

    # Extract labels
    labels = df["label"].to_numpy()

    # Extract pixel values
    pixels = df.drop("label", axis=1).to_numpy(dtype=np.uint8)

    # Convert each row into a 28x28 image
    images = pixels.reshape(-1, 28, 28)

    features = []

    for i, image_array in enumerate(images):

        image = Image.fromarray(image_array).convert("RGB")

        vector = extraction(image)

        features.append(vector)

        if i % 100 == 0:
            print(f"Processed {i}/{len(images)} images")

    return np.array(features), images, labels
def load_or_extract_features(csv_file, limit=1000):

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    tag = str(limit) if limit is not None else "all"

    features_file = FEATURES_DIR / f"fashion_{tag}_features.npy"
    images_file = FEATURES_DIR / f"fashion_{tag}_images.npy"
    labels_file = FEATURES_DIR / f"fashion_{tag}_labels.npy"

    if (features_file.exists()
        and images_file.exists()
        and labels_file.exists()):

        print("Loading saved features...")

        features = np.load(features_file)
        images = np.load(images_file)
        labels = np.load(labels_file)

    else:

        print("Extracting features...")

        features, images, labels = dataset_treatment(csv_file, limit)

        np.save(features_file, features)
        np.save(images_file, images)
        np.save(labels_file, labels)

        print("Features saved!")

    print("Features shape:", features.shape)
    print("Number of images:", len(images))

    return features, images, labels
# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = BASE_DIR / "datasets"
FASHION_DIR = DATASETS_DIR / "FASHION-MNIST"

FASHION_CSV = FASHION_DIR / "fashion-mnist_train.csv"

FEATURES_DIR = BASE_DIR / "features"
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


def search_similar_images(query_index, features, images, top_k=6):

    # Get query image and its features
    query_image = images[query_index]
    query_features = features[query_index]

    # Calculate cosine similarity
    scores = cosine_similarity(
        query_features.reshape(1, -1),
        features
    )[0]

    # Exclude the query image itself
    scores[query_index] = -1

    # Rank images by similarity
    top_indices = np.argsort(scores)[::-1][:top_k]

    return query_image, top_indices, scores

def display_results(query_image, top_indices, scores, images, labels):

    fig, axes = plt.subplots(
        1, len(top_indices) + 1,
        figsize=(15, 4)
    )

    axes = np.atleast_1d(axes)

    axes[0].imshow(query_image, cmap="gray")
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    for position, index in enumerate(top_indices):

        axes[position + 1].imshow(images[index], cmap="gray")

        axes[position + 1].set_title(
            f"Similarity: {scores[index]:.2%}\nClass: {labels[index]}"
        )

        axes[position + 1].axis("off")

    plt.tight_layout()
    plt.show()
    
    # Display results
if __name__ == "__main__":

    # Load or extract features
    features, images, labels = load_or_extract_features(
        FASHION_CSV,
        limit=1000
    )

    # Select query image
    query_index = 0

    # Search
    query_image, top_indices, scores = search_similar_images(
        query_index,
        features,
        images,
        top_k=6
    )

    # Display
    display_results(
        query_image,
        top_indices,
        scores,
        images,
        labels
    )