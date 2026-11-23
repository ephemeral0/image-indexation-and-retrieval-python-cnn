
import numpy as np
import torch
from torchvision import transforms

from modules.config import FEATURES_DIR
from modules.networks import load_network


# Image preprocessing
pretreatment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extraction(image, model):

    image = pretreatment(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features.flatten().numpy()



def load_or_extract_features(
    dataset_name,
    images,
    network_name,
    cache_key=None
):

    # Use a separate cache for each dataset configuration
    if cache_key is None:
        cache_key = dataset_name

    cache_name = f"{cache_key}_{network_name}_{len(images)}"

    features_file = FEATURES_DIR / f"{cache_name}_features.npy"

    # Load existing features if available
    if features_file.exists():

        print("Loading saved features...")
        features = np.load(features_file)

    else:

        print("Extracting features for the first time...")

        model = load_network(network_name)

        features = []

        for i, image in enumerate(images):

            vector = extraction(image, model)
            features.append(vector)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} images")

        features = np.array(features)

        np.save(features_file, features)

        print("Features extracted and saved!")

    print("Features shape:", features.shape)

    return features
