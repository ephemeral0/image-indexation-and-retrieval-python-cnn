import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

#TO DO :
#import image base
#resize images
#extract characteristics vector from images + save results:
## SqueezeNet 1.1 (D = 512) ResNet-18 (D = 512), ResNet-50 (D = 2048), VGG-16 (D = 4096)
#extract CharVec of query image + calculate distance with all the vectors in the base

#Parcourir +ieurs classes similaires//////////////////


model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)

#import image base
path = "/home/leticia/Documents/images"
qIm_path = "/home/leticia/Documents/images/283.jpeg"

base_paths=[]
i = 0
print("\n GOING THROUGH  IMAGES\n")
for img in sorted(os.listdir(path)):
    if img.lower().endswith((".jpg", ".jpeg", ".png")):
        if img.lower().find("283.jpeg")==0:
            print("\n Query image found and SKIPPED\n")
            continue
        full_img_path = os.path.join(path, img)
        base_paths.append(full_img_path)

        i += 1
        if i % 10 == 0:
            print(f"Processed {i}/300images")
        if i == 300:
            break
if len(base_paths) >0:
    print("\n COLLECTING PATHS SUCCEEDED")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_squeezenet_features(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # Shape: (1, 3, 224, 224)

    with torch.no_grad():

        features = model.features(tensor)  # Shape: (1, 512, 13, 13)


        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # Shape: (1, 512, 1, 1)

        # Flatten to 1D array of 512 numbers
        feature_vector = pooled.squeeze().numpy()

        # Normalize vector (L2 norm)
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector /= norm

    return feature_vector  # Shape: (512,)

CharVecs =[]
for p in base_paths:
    CharVecs.append(extract_squeezenet_features(p))

result_matrix = np.array(CharVecs)
print("\nShape of result matrix")
print(result_matrix.shape)
print("\nsize of result matrix")
print(result_matrix.size)
np.savez("features_database.npz", features=result_matrix, paths=np.array(base_paths))

#Query Image Treatment

qIm_features = np.array(extract_squeezenet_features(qIm_path)).reshape(1,-1)

#loading characterics feature vectors

data = np.load("features_database.npz")
db_matrix = data["features"]
db_paths = data["paths"]


from scipy.spatial.distance import cdist
print("\nShape of db matrix")
print(db_matrix.shape)
distances = cdist(qIm_features, result_matrix, metric="cosine")[0]
top_10_indices = np.argsort(distances)[:10]

for rank, idx in enumerate(top_10_indices, start=1):
    print(f"Rank {rank}: Image {db_paths[idx]} | Cosine Distance: {distances[idx]:.5f}")



import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("QtAgg")  # "QtAgg" Or "GTK4Agg" / "TkAgg"

# 1. Create a figure with a grid: 1 query slot + 10 result slots = 11 total slots
# A 3x4 grid gives 12 subplots, perfectly fitting all 11 images
fig, axes = plt.subplots(3, 4, figsize=(12, 12))
axes = axes.flatten()

# 2. Display the Query Image in the first slot
query_img = Image.open(qIm_path).convert("RGB")
axes[0].imshow(query_img)
axes[0].set_title("QUERY IMAGE", fontsize=12, fontweight="bold", color="crimson")
axes[0].axis("off")

# 3. Plot the Top 10 Matched Images
for rank, idx in enumerate(top_10_indices, start=1):
    matched_img_path = base_paths[idx]
    img = Image.open(matched_img_path).convert("RGB")
    ax = axes[rank]
    ax.imshow(img)
    file_name = os.path.basename(matched_img_path)
    score = distances[idx]
    ax.set_title(f"Rank {rank}: {file_name}\nDist: {score:.4f}", fontsize=10)
    ax.axis("off")

# 4. Hide the unused 12th slot in the 3x4 grid
axes[11].axis("off")

plt.tight_layout()
plt.show()


#////////////// RECALL PRECISION CURVES //////////------→--------
