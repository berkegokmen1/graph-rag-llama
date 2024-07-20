from abc import ABC, abstractmethod


class BaseGraphEmbeddingModel(ABC):
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    @abstractmethod
    def process_graph(self, graph, device):
        pass

    @abstractmethod
    def create_model(self, device):
        pass

    @abstractmethod
    def train_model(self, model, device):
        pass

    @abstractmethod
    def generate_embeddings(self, model, device):
        pass
