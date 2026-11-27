import os
import sys
import re
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

##################### NOTES :
# for query 2797 = used classes 27 and 41
# for query 6144 = used classes 61 and 64 and 66
# for query 6640 = used classes 61 and 64 and 66
# for query 6624 and 1447 = used classes 14 and 64 and 66
#####################

# ==============================================================================
MODEL_NAME = "resnet18"  # Use 'resnet18' (D=512) or 'resnet50' (D=2048)
#IMAGES_DIR = "/home/leticia/Documents/images"
#QUERY_IMG_PATH = "/home/leticia/Documents/images/6624.jpeg"
IMAGES_DIR = "/home/leticia/Documents/Uni/M2/IRDM/corel/Corel-10K/64/"
QUERY_IMG_PATH = "/home/leticia/Documents/Uni/M2/IRDM/corel/Corel-10K/64/6421.jpeg"
DB_SAVE_PATH = f"features_database_{MODEL_NAME}.npz"

# Load pre-trained ResNet and replace final classification layer with Identity
# Replace the final 1000-class fully-connected layer with Identity
# so the model outputs the pooled feature vector directly (2048 for ResNet-50, 512 for ResNet-18)
model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==============================================================================
#
# ==============================================================================
def get_corel_class(filepath):
    """Extracts the Corel 10K category ID (0-99) from the numeric filename."""
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1)) // 100
    raise ValueError(f"Could not parse numeric ID from {filename}")

def extract_resnet_features(image_path):
    """Extracts an L2-normalized feature vector from a single image."""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        features = model(tensor)
        feature_vector = features.squeeze().numpy()

        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector /= norm

    return feature_vector

def evaluate_single_query(query_vec, query_class, db_mat, db_cls, metric="cosine"):
    """ Recall, Precision, and Average Precision (AP) for one query."""
    distances = cdist(query_vec.reshape(1, -1), db_mat, metric=metric)[0]
    ranked_indices = np.argsort(distances)

    ranked_classes = db_cls[ranked_indices]
    relevance = (ranked_classes == query_class).astype(np.float32)

    total_relevant = np.sum(relevance)
    if total_relevant == 0:
        return None, None, 0.0

    cum_relevant = np.cumsum(relevance)
    ranks = np.arange(1, len(relevance) + 1)

    precisions = cum_relevant / ranks
    recalls = cum_relevant / total_relevant
    ap = np.sum(precisions * relevance) / total_relevant

    return recalls, precisions, ap


# ==============================================================================
# 3. COLLECT DATABASE IMAGES (BALANCED BY COREL CLASS)
# ==============================================================================
TARGET_CLASSES = [14,64,66]
base_paths = []
query_filename = os.path.basename(QUERY_IMG_PATH).lower()

print("\n--- Collecting Balanced Image Paths ---")
all_files = sorted(os.listdir(IMAGES_DIR))
for img in all_files:
    if img.lower().endswith((".jpg", ".jpeg", ".png")):
        if img.lower() == query_filename:
            print(f"Query image ({img}) found and skipped from database.")
            continue
        full_path = os.path.join(IMAGES_DIR, img)
        if get_corel_class(full_path) in TARGET_CLASSES:
            base_paths.append(full_path)
if len(base_paths) == 0:
    print(f"Error: No database images found for classes {TARGET_CLASSES} in '{IMAGES_DIR}'.")
    print(f"Note: Query image '{os.path.basename(QUERY_IMG_PATH)}' belongs to Class {get_corel_class(QUERY_IMG_PATH)}.")
    sys.exit(1)
else:
    print(f"Collected {len(base_paths)} database images across classes {TARGET_CLASSES}.")


# ==============================================================================
# 4. EXTRACT & SAVE FEATURE VECTORS
# ==============================================================================
print("\n--- Extracting ResNet Feature Vectors ---")
char_vecs = []
for idx, p in enumerate(base_paths, start=1):
   char_vecs.append(extract_resnet_features(p))
   if idx % 25 == 0 or idx == len(base_paths):
        print(f"Processed {idx}/{len(base_paths)} images")

result_matrix = np.array(char_vecs)
np.savez(DB_SAVE_PATH, features=result_matrix, paths=np.array(base_paths))
print(f"Saved database matrix of shape {result_matrix.shape} to '{DB_SAVE_PATH}'.")



# ==============================================================================
# 5. QUERY RETRIEVAL (TOP-10 COSINE SIMILARITY)
# ==============================================================================
data = np.load(DB_SAVE_PATH)
db_matrix = data["features"]
db_paths = data["paths"]
db_classes = np.array([get_corel_class(p) for p in db_paths])

qIm_features = extract_resnet_features(QUERY_IMG_PATH).reshape(1, -1)
query_class = get_corel_class(QUERY_IMG_PATH)

distances = cdist(qIm_features, db_matrix, metric="cosine")[0]
top_10_indices = np.argsort(distances)[:10]

print(f"\n--- Top 10 Matches for {os.path.basename(QUERY_IMG_PATH)} (Class {query_class}) ---")
for rank, idx in enumerate(top_10_indices, start=1):
    print(f"Rank {rank:2d}: {os.path.basename(db_paths[idx])} (Class {db_classes[idx]}) | Cosine Dist: {distances[idx]:.5f}")


# ==============================================================================
# 6. PRECISION-RECALL & MEAN AVERAGE PRECISION (mAP) EVALUATION
# ==============================================================================
# A. Single Query P-R Curve
q_recalls, q_precisions, single_ap = evaluate_single_query(
    qIm_features, query_class, db_matrix, db_classes, metric="cosine"
)

all_aps = []
recall_levels = np.linspace(0.0, 1.0, 11)
interpolated_precisions = []

for i in range(len(db_matrix)):
    mask = np.ones(len(db_matrix), dtype=bool)
    mask[i] = False  # Exclude query itself

    rec, prec, ap = evaluate_single_query(
        db_matrix[i], db_classes[i], db_matrix[mask], db_classes[mask], metric="cosine"
    )
    if rec is not None:
        all_aps.append(ap)
        interp_p = [np.max(prec[rec >= r]) if np.any(rec >= r) else 0.0 for r in recall_levels]
        interpolated_precisions.append(interp_p)

mAP = np.mean(all_aps)
mean_pr_curve = np.mean(interpolated_precisions, axis=0)

print("\n==================================================")
print(f"Single Query Average Precision (AP) : {single_ap:.4f} ({single_ap*100:.2f}%)")
print(f"Database Mean Average Precision (mAP): {mAP:.4f} ({mAP*100:.2f}%)")
print("==================================================")


# ==============================================================================
# 7. VISUALIZE TOP-10 RETRIEVAL & EVALUATION CURVES
# ==============================================================================
# Window 1: Query Image + Top 10 Matches
fig1, axes1 = plt.subplots(3, 4, figsize=(12, 10))
axes1 = axes1.flatten()

axes1[0].imshow(Image.open(QUERY_IMG_PATH).convert("RGB"))
axes1[0].set_title(f"QUERY (Class {query_class})\n{os.path.basename(QUERY_IMG_PATH)}", fontsize=11, fontweight="bold", color="crimson")
axes1[0].axis("off")

for rank, idx in enumerate(top_10_indices, start=1):
    matched_path = db_paths[idx]
    ax = axes1[rank]
    ax.imshow(Image.open(matched_path).convert("RGB"))
    is_match = "✓" if db_classes[idx] == query_class else "✗"
    ax.set_title(f"Rank {rank} [{is_match}]: {os.path.basename(matched_path)}\nDist: {distances[idx]:.4f}", fontsize=9)
    ax.axis("off")

axes1[11].axis("off")
fig1.suptitle(f"Top-10 Retrieval Results ({MODEL_NAME.upper()})", fontsize=14, fontweight="bold")
plt.tight_layout()

# Window 2: Precision-Recall Curves (Single Query & Overall 11-Point mAP)
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

axes2[0].plot(q_recalls, q_precisions, color="navy", lw=2, label=f"AP = {single_ap:.4f}")
axes2[0].set_title(f"Precision-Recall Curve: {os.path.basename(QUERY_IMG_PATH)} (Class {query_class})")
axes2[0].set_xlabel("Recall")
axes2[0].set_ylabel("Precision")
axes2[0].set_xlim([0.0, 1.0])
axes2[0].set_ylim([0.0, 1.05])
axes2[0].grid(alpha=0.3)
axes2[0].legend(loc="upper right")

axes2[1].plot(recall_levels, mean_pr_curve, marker="o", color="crimson", lw=2, label=f"mAP = {mAP:.4f}")
axes2[1].set_title(f"Mean 11-Point Precision-Recall Curve ({MODEL_NAME.upper()})")
axes2[1].set_xlabel("Recall")
axes2[1].set_ylabel("Mean Precision")
axes2[1].set_xlim([0.0, 1.0])
axes2[1].set_ylim([0.0, 1.05])
axes2[1].grid(alpha=0.3)
axes2[1].legend(loc="upper right")

plt.tight_layout()
plt.show()
