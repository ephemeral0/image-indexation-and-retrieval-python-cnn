
import pickle
import numpy as np
import pandas as pd

from PIL import Image
from modules.config import DATASET_PATHS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

FASHION_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


# =====================================
# 1. COREL
# =====================================

def load_corel(dataset_name):

    directory = DATASET_PATHS[dataset_name]

    if not directory.exists():
        raise FileNotFoundError(f"Dataset not found: {directory}")

    images = []
    image_ids = []
    classes = []

    image_files = sorted(
        file for file in directory.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    for file in image_files:

        image = Image.open(file).convert("RGB")
        images.append(image)

        # Generate unique image IDs
        if dataset_name == "Corel-1K":
            image_ids.append(file.name)
            classes.append(None)

        else:
            image_ids.append(str(file.relative_to(directory)))

            # Class is determined by the parent folder
            classes.append(file.parent.name)

    print(f"Dataset: {dataset_name}")
    print(f"Images loaded: {len(images)}")

    return images, image_ids, classes


# =====================================
# 2. CIFAR-100
# =====================================

def load_cifar(split="train", limit=None):

    directory = DATASET_PATHS["CIFAR-100"]
    file_path = directory / split
    meta_path = directory / "meta"

    with open(file_path, "rb") as file:
        data = pickle.load(file, encoding="bytes")

    with open(meta_path, "rb") as file:
        meta = pickle.load(file, encoding="bytes")

    pixels = data[b"data"]
    labels = data[b"fine_labels"]
    class_names = meta[b"fine_label_names"]

    images = pixels.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    pil_images = [
        Image.fromarray(array).convert("RGB")
        for array in images
    ]

    image_ids = [
        f"cifar_{split}_{i:06d}"
        for i in range(len(pil_images))
    ]

    classes = [
        class_names[int(label)].decode("utf-8")
        if isinstance(class_names[int(label)], bytes)
        else class_names[int(label)]
        for label in labels
    ]

    print(f"Dataset: CIFAR-100 ({split})")
    print(f"Images loaded: {len(pil_images)}")

    return pil_images, image_ids, classes


# =====================================
# 3. FASHION-MNIST
# =====================================

def load_fashion(split="train", limit=None):

    directory = DATASET_PATHS["Fashion-MNIST"]
    csv_path = directory / f"fashion-mnist_{split}.csv"

    df = pd.read_csv(csv_path)

    if limit is not None:
        df = df.iloc[:limit]

    labels = df["label"].to_numpy()
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.uint8)

    image_arrays = pixels.reshape(-1, 28, 28)

    pil_images = [
        Image.fromarray(array).convert("RGB")
        for array in image_arrays
    ]

    image_ids = [
        f"fashion_{split}_{i:06d}"
        for i in range(len(pil_images))
    ]

    classes = [
        FASHION_CLASSES[int(label)]
        for label in labels
    ]

    print(f"Dataset: Fashion-MNIST ({split})")
    print(f"Images loaded: {len(pil_images)}")

    return pil_images, image_ids, classes
