from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"

class Embedder:
    def __init__(self, model_name: str = MPNET_BASE_V2):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]):
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        return self.embeddings

    def top_n(self, text, n) -> (np.array, np.array):
        embedding = self.model.encode([text])
        # print("get_n_closest_batch")
        # expanded_embedding = np.expand_dims(embedding, axis=0)
        expanded_embedding = embedding
        # for matrix multiplication we need the shape to be NXM MXN vector 1X384, matrix 384X3
        angles = cosine_similarity(expanded_embedding, self.embeddings).squeeze()
        # https://stackoverflow.com/a/6910672
        # ::-1 reverses this list, # -n: top N
        sorted_indices = angles.argsort()[-n:][::-1]
        # print(f"sorted indices: {sorted_indices}")
        return sorted_indices, angles[sorted_indices]