from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
import numpy as np


class PrecomputedEmbedding(BaseEmbedding):
    _embeddings = PrivateAttr()
    _node_ids = PrivateAttr()

    def __init__(self, embeddings):
        super().__init__()
        self._embeddings = embeddings
        self._node_ids = list(embeddings.keys())

    def _get_query_embedding(self, query: str):
        # For queries, we'll use the average of all embeddings
        # This is a simple approach and might not be optimal for all use cases
        all_embeddings = np.array(list(self._embeddings.values()))
        return np.mean(all_embeddings, axis=0).tolist()

    def _aget_query_embedding(self, query: str):
        # This method won't be used in our case.
        raise NotImplementedError("Async query embedding not supported for precomputed embeddings")

    def _get_text_embedding(self, text: str):
        # Assume the text is the node ID
        node_id = int(text.split(":")[1].split(",")[0].strip())
        return self._embeddings[node_id].tolist()

    def embed_documents(self, documents):
        embedded_documents = []
        for doc in documents:
            node_id = int(doc.split(":")[1].split(",")[0].strip())
            if node_id in self._embeddings:
                embedded_documents.append(self._embeddings[node_id].tolist())
            else:
                # If we don't have an embedding for this node, use a zero vector
                # You might want to handle this case differently
                embedded_documents.append([0.0] * len(next(iter(self._embeddings.values()))))
        return embedded_documents
