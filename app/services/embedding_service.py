import numpy as np


def get_embeddings(texts):

    embeddings = []

    for _ in texts:

        vec = np.random.rand(384)

        embeddings.append(
            vec.tolist()
        )

    return embeddings