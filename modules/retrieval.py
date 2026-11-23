
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def search_similar_images(
    query_id,
    image_ids,
    features,
    top_k=10
):

    # Find the query image
    if query_id not in image_ids:
        raise ValueError(f"Image not found: {query_id}")

    query_index = image_ids.index(query_id)

    # Get its already-extracted features
    query_features = features[query_index]

    # Calculate cosine similarity
    scores = cosine_similarity(
        query_features.reshape(1, -1),
        features
    )[0]

    # Exclude the query image itself
    scores[query_index] = -1

    # Get the most similar images
    top_indices = np.argsort(scores)[::-1][:top_k]

    return query_index, top_indices, scores
