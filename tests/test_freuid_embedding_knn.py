from __future__ import annotations

import numpy as np
import torch

from scripts.run_freuid_embedding_knn import _knn_scores


def test_embedding_knn_scores_are_weighted_label_probabilities() -> None:
    train_embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    train_embeddings /= np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_labels = np.asarray([1, 1, 0], dtype=np.int64)
    query_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    scores = _knn_scores(
        train_embeddings,
        train_labels,
        query_embeddings,
        k_values=[1, 3],
        temperatures=[10.0],
        query_batch_size=2,
        device=torch.device("cpu"),
    )

    assert scores[(1, 10.0)].tolist() == [1.0, 0.0]
    assert scores[(3, 10.0)][0] > 0.99
    assert scores[(3, 10.0)][1] < 0.01
