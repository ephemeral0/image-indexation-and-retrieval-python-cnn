
import pickle
from pathlib import Path

from PIL import Image
from torchvision import transforms
import torch
import torchvision.models as models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"

CIFAR_DIR = DATASETS_DIR / "Cifar"
CIFAR_TRAIN_FILE = CIFAR_DIR / "train"

FEATURES_DIR = BASE_DIR / "features"


# =====================================
# 2. IMAGE PREPROCESSING
# =====================================

pretreatment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# 3. RESNET-50 MODEL
# =====================================

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Remove the classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()


# =====================================
# 4. LOAD CIFAR-100
# =====================================

def load_cifar100(file_path, limit=1000):

    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="bytes")

    pixels = data[b"data"]
    labels = data[b"fine_labels"]

    # Reconstruct RGB images
    images = pixels.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    return images, np.array(labels)


# =====================================
# 5. FEATURE EXTRACTION
# =====================================

def extraction(image):

    image = pretreatment(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features.flatten().numpy()


def dataset_treatment(images):

    features = []

    for i, image_array in enumerate(images):

        image = Image.fromarray(image_array)

        vector = extraction(image)
        features.append(vector)

        if i % 100 == 0:
            print(f"Processed {i}/{len(images)} images")

    return np.array(features)


# =====================================
# 6. LOAD OR EXTRACT FEATURES
# =====================================

def load_or_extract_features(file_path, limit=1000):

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    tag = str(limit) if limit is not None else "all"

    features_file = FEATURES_DIR / f"cifar100_train_{tag}_features.npy"
    images_file = FEATURES_DIR / f"cifar100_train_{tag}_images.npy"
    labels_file = FEATURES_DIR / f"cifar100_train_{tag}_labels.npy"

    if (features_file.exists()
            and images_file.exists()
            and labels_file.exists()):

        print("Loading saved features...")

        features = np.load(features_file)
        images = np.load(images_file)
        labels = np.load(labels_file)

    else:

        print("Extracting features for the first time...")

        images, labels = load_cifar100(file_path, limit)

        features = dataset_treatment(images)

        np.save(features_file, features)
        np.save(images_file, images)
        np.save(labels_file, labels)

        print("Features extracted and saved!")

    print("Features shape:", features.shape)
    print("Number of images:", len(images))

    return features, images, labels


# =====================================
# 7. INSPECT DATASET
# =====================================

def inspect_dataset(name, directory):

    print(f"\n{'=' * 40}")
    print(f"DATASET: {name}")
    print(f"{'=' * 40}")

    if not directory.exists():
        print("Directory not found!")
        return

    files = [f for f in directory.rglob("*") if f.is_file()]

    print(f"Total files: {len(files)}")

    for file in files:
        print(file.relative_to(directory))


# =====================================
# 8. SEARCH SIMILAR IMAGES
# =====================================

def search_similar_images(query_index, features, images, top_k=6):

    query_image = images[query_index]
    query_features = features[query_index]

    scores = cosine_similarity(
        query_features.reshape(1, -1),
        features
    )[0]

    # Exclude query image itself
    scores[query_index] = -1

    # Sort from highest similarity to lowest
    top_indices = np.argsort(scores)[::-1][:top_k]

    return query_image, top_indices, scores


# =====================================
# 9. DISPLAY RESULTS
# =====================================

def display_results(query_image, top_indices, scores, images, labels):

    fig, axes = plt.subplots(
        1, len(top_indices) + 1,
        figsize=(15, 4)
    )

    axes = np.atleast_1d(axes)

    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    for position, index in enumerate(top_indices):

        axes[position + 1].imshow(images[index])

        axes[position + 1].set_title(
            f"Similarity: {scores[index]:.2%}\nClass: {labels[index]}"
        )

        axes[position + 1].axis("off")

    plt.tight_layout()
    plt.show()


# =====================================
# 10. MAIN
# =====================================

if __name__ == "__main__":

    inspect_dataset("CIFAR-100", CIFAR_DIR)

    features, images, labels = load_or_extract_features(
        CIFAR_TRAIN_FILE,
        limit=1000
    )

    query_index = 0

    query_image, top_indices, scores = search_similar_images(
        query_index,
        features,
        images,
        top_k=6
    )

    display_results(
        query_image,
        top_indices,
        scores,
        images,
        labels
    )
