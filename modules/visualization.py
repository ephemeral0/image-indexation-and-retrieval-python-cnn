
import matplotlib.pyplot as plt


def display_results(
    images,
    image_ids,
    classes,
    query_index,
    top_indices,
    scores
):

    # Create a grid: 3 rows × 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()

    # Display the query image
    axes[0].imshow(images[query_index])
    axes[0].set_title(
        f"QUERY IMAGE\n{image_ids[query_index]}",
        fontsize=10,
        fontweight="bold"
    )
    axes[0].axis("off")

    # Display the retrieved images
    for position, index in enumerate(top_indices, start=1):

        image_class = classes[index]

        if image_class is None:
            image_class = "Not available"

        axes[position].imshow(images[index])

        axes[position].set_title(
            f"Rank {position}\n"
            f"ID: {image_ids[index]}\n"
            f"Class: {image_class}\n"
            f"Similarity: {scores[index]:.2%}",
            fontsize=9
        )

        axes[position].axis("off")

    # Hide the unused space
    for position in range(len(top_indices) + 1, len(axes)):
        axes[position].axis("off")

    plt.tight_layout()
    plt.show()
