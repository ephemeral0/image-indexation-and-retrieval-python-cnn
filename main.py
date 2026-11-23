from pathlib import Path
from modules.data_loader import (
    load_corel,
    load_cifar,
    load_fashion
)

from modules.feature_extraction import load_or_extract_features
from modules.retrieval import search_similar_images
from modules.visualization import display_results

from modules.config import TOP_K


# =====================================
# 1. DATASET SELECTION
# =====================================

print("\n===== IMAGE RETRIEVAL SYSTEM =====")

print("\nChoose a dataset:")
print("1. Corel")
print("2. CIFAR-100")
print("3. Fashion-MNIST")

dataset_choice = input("\nYour choice: ")

if dataset_choice == "1":

    print("\nChoose Corel version:")
    print("1. Corel-1K")
    print("2. Corel-5K")
    print("3. Corel-10K")

    version = input("\nYour choice: ")

    corel_versions = {
        "1": "Corel-1K",
        "2": "Corel-5K",
        "3": "Corel-10K"
    }

    if version not in corel_versions:
        raise ValueError("Invalid Corel version.")

    dataset_name = corel_versions[version]

    images, image_ids, classes = load_corel(dataset_name)

    cache_key = dataset_name


elif dataset_choice == "2":

    dataset_name = "CIFAR-100"

    print("\nChoose split:")
    print("1. Train")
    print("2. Test")

    split_choice = input("\nYour choice: ")

    if split_choice not in ["1", "2"]:
        raise ValueError("Invalid split.")

    split = "train" if split_choice == "1" else "test"

    images, image_ids, classes = load_cifar(split=split)

    cache_key = f"CIFAR-100_{split}"


elif dataset_choice == "3":

    dataset_name = "Fashion-MNIST"

    print("\nChoose split:")
    print("1. Train")
    print("2. Test")

    split_choice = input("\nYour choice: ")

    if split_choice not in ["1", "2"]:
        raise ValueError("Invalid split.")

    split = "train" if split_choice == "1" else "test"

    images, image_ids, classes = load_fashion(split=split)

    cache_key = f"Fashion-MNIST_{split}"


else:
    raise ValueError("Invalid dataset choice.")


# =====================================
# 2. NETWORK SELECTION
# =====================================

print("\nChoose a network:")
print("1. ResNet-18")
print("2. ResNet-50")
print("3. VGG-16")

network_choice = input("\nYour choice: ")

networks = {
    "1": "ResNet-18",
    "2": "ResNet-50",
    "3": "VGG-16"
}

if network_choice not in networks:
    raise ValueError("Invalid network choice.")

network_name = networks[network_choice]


# =====================================
# 3. FEATURE EXTRACTION
# =====================================

features = load_or_extract_features(
    dataset_name,
    images,
    network_name,
    cache_key=cache_key
)


# =====================================
# 4. QUERY IMAGE
# =====================================

print("\nDataset loaded successfully.")
print(f"Total images: {len(images)}")

print("\nExample image IDs:")
for image_id in image_ids[:5]:
    print("-", image_id)

query_number = input("\nEnter the query image number: ").strip()

if dataset_name == "Corel-1K":

    query_id = next(
        (img_id for img_id in image_ids
         if Path(img_id).stem == str(int(query_number))),
        None
    )

elif dataset_name in ["Corel-5K", "Corel-10K"]:

    folder_number = int(query_number) // 100

    query_id = next(
        (img_id for img_id in image_ids
         if Path(img_id).parent.name == str(folder_number)
         and Path(img_id).stem == query_number),
        None
    )

elif dataset_name == "CIFAR-100":

    query_id = f"cifar_{split}_{int(query_number):06d}"

elif dataset_name == "Fashion-MNIST":

    query_id = f"fashion_{split}_{int(query_number):06d}"

if query_id is None or query_id not in image_ids:
    raise ValueError(f"Image number {query_number} not found.")

# =====================================
# 5. IMAGE RETRIEVAL
# =====================================

query_index, top_indices, scores = search_similar_images(
    query_id,
    image_ids,
    features,
    top_k=TOP_K
)


# =====================================
# 6. DISPLAY RESULTS
# =====================================

print("\nTop 10 similar images:\n")

for rank, index in enumerate(top_indices, start=1):

    image_class = classes[index]

    if image_class is None:
        image_class = "Not available"

    print(
        f"{rank}. {image_ids[index]} "
        f"| Similarity: {scores[index]:.2%}"
    )

display_results(
    images,
    image_ids,
    classes,
    query_index,
    top_indices,
    scores
)
