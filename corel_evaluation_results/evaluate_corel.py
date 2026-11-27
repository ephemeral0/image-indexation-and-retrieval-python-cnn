"""
Class-based evaluation of CNN feature retrieval on Corel (1K / 5K / 10K).

For every experiment:
  - pick N random classes,
  - pick M random query images inside each chosen class,
  - retrieve the TOP_K (10) closest images with cosine distance (query excluded),
  - an image is relevant if it belongs to the SAME class (folder) as the query,
  - compute Precision, Recall, F-score per query,
  - average per class, then average over classes -> mAP, mAR, mAF.

Query vectors are taken directly from the saved .npz database (they are the same
vectors that extract_features_and_labels() would produce), so no network has to
be loaded and the evaluation is fast.
"""
import os
import json
import numpy as np
import pandas as pd

# ==============================================================================
# 1. CONFIGURATION  (edit the paths below)
# ==============================================================================
PROJECT_DIR = "/home/leticia/Documents/Uni/M2/IRDM/project/"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")

TOP_K = 10
SEED = 42  # same seed -> same queries for every network (fair comparison)

# One entry per Corel base. Adjust the .npz file names of 5K / 10K to yours.
EXPERIMENTS = [
    {
        "name": "Corel-1K",
        "n_classes": 5,
        "n_queries": 10,
        "dbs": {
            "resnet18": PROJECT_DIR + "corel_features_resnet18.npz",
            "squeezenet1_1": PROJECT_DIR + "corel_features_squeezenet1_1.npz",
        },
    },
    {
        "name": "Corel-5K",
        "n_classes": 20,
        "n_queries": 15,
        "dbs": {
            "resnet18": PROJECT_DIR + "corel_features_resnet18.npz",
            "squeezenet1_1": PROJECT_DIR + "corel_features_squeezenet1_1.npz",
        },
    },
    {
        "name": "Corel-10K",
        "n_classes": 30,
        "n_queries": 20,
        "dbs": {
            "resnet18": PROJECT_DIR + "corel_features_resnet18.npz",
            "squeezenet1_1": PROJECT_DIR + "corel_features_squeezenet1_1.npz", ## add vgg
        },
    },
]


# ==============================================================================
# 2. HELPERS
# ==============================================================================
def load_db(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        feats = data["features"].astype(np.float32)
        paths = np.array([str(p) for p in data["paths"]])
        classes = np.array([str(c) for c in data["class_folders"]])
    # re-normalise (harmless if already L2-normalised) -> cosine = dot product
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feats / norms, paths, classes


def select_queries(paths, classes, n_classes, n_queries, rng):
    """Returns {class_name: [path, ...]} using sorted paths so the selection is
    reproducible and identical across networks."""
    all_classes = sorted(set(classes))
    if n_classes > len(all_classes):
        raise ValueError(f"Requested {n_classes} classes but only {len(all_classes)} exist.")
    chosen = rng.choice(all_classes, size=n_classes, replace=False)

    selection = {}
    for c in sorted(chosen):
        class_paths = sorted(paths[classes == c])
        k = min(n_queries, len(class_paths))
        if k < n_queries:
            print(f"[WARN] class '{c}' has only {len(class_paths)} images (< {n_queries}).")
        selection[c] = list(rng.choice(class_paths, size=k, replace=False))
    return selection


def prf(n_relevant_retrieved, k, n_relevant_total):
    precision = n_relevant_retrieved / k
    recall = n_relevant_retrieved / n_relevant_total if n_relevant_total > 0 else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f


def run_experiment(exp, model_name, npz_path):
    feats, paths, classes = load_db(npz_path)
    path_to_idx = {p: i for i, p in enumerate(paths)}
    class_sizes = {c: int((classes == c).sum()) for c in set(classes)}

    # same RNG seed for every network -> same queries
    rng = np.random.default_rng(SEED)
    selection = select_queries(paths, classes, exp["n_classes"], exp["n_queries"], rng)

    rows = []
    for cls, query_paths in selection.items():
        for qp in query_paths:
            qi = path_to_idx[qp]
            sims = feats @ feats[qi]
            dist = 1.0 - sims
            dist[qi] = np.inf  # exclude the query itself
            top = np.argsort(dist)[:TOP_K]

            hits = int((classes[top] == cls).sum())
            n_rel_total = class_sizes[cls] - 1  # relevant images excluding the query
            p, r, f = prf(hits, TOP_K, n_rel_total)
            rows.append({
                "dataset": exp["name"], "model": model_name, "class": cls,
                "query": os.path.basename(qp), "hits@10": hits,
                "precision": p, "recall": r, "f_score": f,
            })
    return pd.DataFrame(rows)


# ==============================================================================
# 3. MAIN
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    per_query_all, per_class_all, summary_rows = [], [], []
    report_lines = []

    def out(line=""):
        print(line)
        report_lines.append(line)

    for exp in EXPERIMENTS:
        for model_name, npz_path in exp["dbs"].items():
            if not os.path.exists(npz_path):
                out(f"[SKIP] {exp['name']} / {model_name}: '{npz_path}' not found.")
                continue

            df = run_experiment(exp, model_name, npz_path)
            per_query_all.append(df)

            # average per class
            per_class = (df.groupby("class")[["precision", "recall", "f_score"]]
                           .mean().reset_index())
            per_class.insert(0, "model", model_name)
            per_class.insert(0, "dataset", exp["name"])
            per_class_all.append(per_class)

            # mean over classes
            m = per_class[["precision", "recall", "f_score"]].mean()
            summary_rows.append({
                "dataset": exp["name"], "model": model_name,
                "n_classes": df["class"].nunique(), "n_queries": len(df),
                "mAP": m["precision"], "mAR": m["recall"], "mAF": m["f_score"],
            })

            out("=" * 78)
            out(f"{exp['name']} | {model_name} | {df['class'].nunique()} classes x "
                f"{exp['n_queries']} queries (Top-{TOP_K})")
            out("=" * 78)
            out(per_class[["class", "precision", "recall", "f_score"]]
                .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            out("-" * 78)
            out(f"mAP = {m['precision']:.4f} | mAR = {m['recall']:.4f} | mAF = {m['f_score']:.4f}")
            out()

    if not summary_rows:
        print("Nothing evaluated - check the .npz paths in EXPERIMENTS.")
        return

    summary = pd.DataFrame(summary_rows)
    out("#" * 78)
    out("GLOBAL SUMMARY")
    out("#" * 78)
    out(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- save everything -----------------------------------------------------
    pd.concat(per_query_all).to_csv(os.path.join(OUTPUT_DIR, "per_query_results.csv"), index=False)
    pd.concat(per_class_all).to_csv(os.path.join(OUTPUT_DIR, "per_class_results.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    with open(os.path.join(OUTPUT_DIR, "summary_results.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
