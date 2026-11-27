"""
Class-based evaluation of CNN feature retrieval on Corel (1K / 5K / 10K).

Each network has ONE .npz containing the features of every Corel image (1K, 5K and
10K together). For each experiment the script first restricts the database to the
images of the requested Corel base (see "subset" in EXPERIMENTS), then:
  - picks N random classes and M random query images per class,
  - retrieves the TOP_K closest images (cosine distance, query excluded) from the
    restricted database only,
  - an image is relevant if it belongs to the SAME class as the query,
  - computes Precision, Recall, F-score per query, averages per class, then over
    classes -> mAP, mAR, mAF.

Run `python evaluate_corel.py --inspect` first to see how the images in your .npz
are organised, then fill in the "subset" rules below.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
PROJECT_DIR = "/home/leticia/Documents/Uni/M2/IRDM/project/"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")

# One npz per network, containing ALL Corel images (1K + 5K + 10K)
DB_FILES = {
    "resnet18":      PROJECT_DIR + "corel_features_resnet18.npz",
    "squeezenet1_1": PROJECT_DIR + "corel_features_squeezenet1_1.npz",
    "resnet50":      PROJECT_DIR + "corel_features_resnet50.npz",
    "vgg16":         PROJECT_DIR + "corel_features_vgg16.npz",
}

TOP_K = 10
SEED = 42  # same seed -> same queries for every network

# "subset" tells the script which images of the npz belong to that Corel base:
#   {"path_contains": "corel1k"}     -> keep images whose path contains this text
#                                       (a list means: any of them; case-insensitive)
#   {"classes": ["0", "1", ...]}      -> keep images whose class folder is in this list
#   None                              -> use every image in the npz
# Use --inspect to see the folder structure and choose the right rule.
EXPERIMENTS = [
    {"name": "Corel-1K",  "n_classes": 5,  "n_queries": 10, "subset": {"path_contains": "1k"}},
    {"name": "Corel-5K",  "n_classes": 20, "n_queries": 15, "subset": {"path_contains": "5k"}},
    {"name": "Corel-10K", "n_classes": 30, "n_queries": 20, "subset": {"path_contains": "10k"}},
]


# ==============================================================================
# 2. HELPERS
# ==============================================================================
def load_db(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        feats = data["features"].astype(np.float32)
        paths = np.array([str(p) for p in data["paths"]])
        classes = np.array([str(c) for c in data["class_folders"]])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feats / norms, paths, classes


def apply_subset(feats, paths, classes, subset):
    """Restricts the database to one Corel base. Retrieval and class sizes are then
    computed only inside this subset."""
    if subset is None:
        return feats, paths, classes
    mask = np.ones(len(paths), dtype=bool)
    if "path_contains" in subset:
        keys = subset["path_contains"]
        keys = [keys] if isinstance(keys, str) else keys
        low = np.char.lower(paths)
        mask &= np.any([np.char.find(low, k.lower()) >= 0 for k in keys], axis=0)
    if "classes" in subset:
        mask &= np.isin(classes, [str(c) for c in subset["classes"]])
    return feats[mask], paths[mask], classes[mask]


def select_queries(paths, classes, n_classes, n_queries, rng):
    all_classes = sorted(set(classes))
    if n_classes > len(all_classes):
        raise ValueError(f"Requested {n_classes} classes but the subset only has {len(all_classes)}.")
    chosen = rng.choice(all_classes, size=n_classes, replace=False)
    selection = {}
    for c in sorted(chosen):
        class_paths = sorted(paths[classes == c])
        k = min(n_queries, len(class_paths))
        if k < n_queries:
            print(f"[WARN] class '{c}' has only {len(class_paths)} images (< {n_queries}).")
        selection[c] = list(rng.choice(class_paths, size=k, replace=False))
    return selection


def prf(hits, k, n_relevant_total):
    precision = hits / k
    recall = hits / n_relevant_total if n_relevant_total > 0 else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f


def run_experiment(exp, model_name, feats, paths, classes):
    feats, paths, classes = apply_subset(feats, paths, classes, exp["subset"])
    if len(paths) == 0:
        raise ValueError(f"{exp['name']}: the subset rule matched no image. Check it with --inspect.")

    path_to_idx = {p: i for i, p in enumerate(paths)}
    class_sizes = {c: int((classes == c).sum()) for c in set(classes)}

    rng = np.random.default_rng(SEED)
    selection = select_queries(paths, classes, exp["n_classes"], exp["n_queries"], rng)

    rows = []
    for cls, query_paths in selection.items():
        for qp in query_paths:
            qi = path_to_idx[qp]
            dist = 1.0 - feats @ feats[qi]
            dist[qi] = np.inf  # exclude the query itself
            top = np.argsort(dist)[:TOP_K]

            hits = int((classes[top] == cls).sum())
            p, r, f = prf(hits, TOP_K, class_sizes[cls] - 1)
            rows.append({
                "dataset": exp["name"], "model": model_name, "class": cls,
                "query": os.path.basename(qp), "hits@10": hits,
                "precision": p, "recall": r, "f_score": f,
            })
    return pd.DataFrame(rows), len(paths)


def inspect():
    """Prints how the images of each npz are organised, to help write the subset rules."""
    for model_name, npz_path in DB_FILES.items():
        if not os.path.exists(npz_path):
            print(f"[SKIP] {model_name}: '{npz_path}' not found.")
            continue
        feats, paths, classes = load_db(npz_path)
        print(f"\n=== {model_name}: {len(paths)} vectors, dim {feats.shape[1]}, "
              f"{len(set(classes))} distinct class folders ===")
        print("Example paths:")
        for p in paths[:: max(1, len(paths) // 5)][:5]:
            print("   ", p)
        parents = pd.Series([os.path.basename(os.path.dirname(os.path.dirname(p))) for p in paths])
        print("Images per grand-parent folder (usually the Corel base):")
        print(parents.value_counts().to_string())
        print("Class folders (first 20):", sorted(set(classes))[:20])
        break  # same structure for every network


# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    per_query_all, per_class_all, summary_rows, report_lines = [], [], [], []

    def out(line=""):
        print(line)
        report_lines.append(line)

    for model_name, npz_path in DB_FILES.items():
        if not os.path.exists(npz_path):
            out(f"[SKIP] {model_name}: '{npz_path}' not found.")
            continue

        feats, paths, classes = load_db(npz_path)  # loaded ONCE per network
        out(f"\n##### {model_name}: {len(paths)} vectors, D = {feats.shape[1]} #####")

        for exp in EXPERIMENTS:
            df, db_size = run_experiment(exp, model_name, feats, paths, classes)
            per_query_all.append(df)

            per_class = df.groupby("class")[["precision", "recall", "f_score"]].mean().reset_index()
            per_class.insert(0, "model", model_name)
            per_class.insert(0, "dataset", exp["name"])
            per_class_all.append(per_class)

            m = per_class[["precision", "recall", "f_score"]].mean()
            summary_rows.append({
                "dataset": exp["name"], "model": model_name, "db_size": db_size,
                "n_classes": df["class"].nunique(), "n_queries": len(df),
                "mAP": m["precision"], "mAR": m["recall"], "mAF": m["f_score"],
            })

            out("=" * 78)
            out(f"{exp['name']} | {model_name} | database: {db_size} images | "
                f"{df['class'].nunique()} classes x {exp['n_queries']} queries (Top-{TOP_K})")
            out("=" * 78)
            out(per_class[["class", "precision", "recall", "f_score"]]
                .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            out("-" * 78)
            out(f"mAP = {m['precision']:.4f} | mAR = {m['recall']:.4f} | mAF = {m['f_score']:.4f}")
            out()

    if not summary_rows:
        print("Nothing evaluated - check DB_FILES.")
        return

    summary = pd.DataFrame(summary_rows)
    out("#" * 78)
    out("GLOBAL SUMMARY")
    out("#" * 78)
    out(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    pd.concat(per_query_all).to_csv(os.path.join(OUTPUT_DIR, "per_query_results.csv"), index=False)
    pd.concat(per_class_all).to_csv(os.path.join(OUTPUT_DIR, "per_class_results.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    with open(os.path.join(OUTPUT_DIR, "summary_results.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    if "--inspect" in sys.argv:
        inspect()
    else:
        main()
