import torch
from torch_geometric.nn import DeepGraphInfomax
from src.models.encoder import GCNEncoder
from src.models.base import BaseGraphEmbeddingModel


class DeepGraphInfomaxModel(BaseGraphEmbeddingModel):
    def process_graph(self, graph, device):
        self.edge_index = torch.tensor(list(graph.edges())).t().contiguous().to(device)
        self.features = torch.eye(graph.number_of_nodes()).to(device)  # One-hot encoding

    def create_model(self, device):
        input_dim = self.features.shape[1]
        return DeepGraphInfomax(
            hidden_channels=self.embedding_dim,
            encoder=GCNEncoder(input_dim, self.embedding_dim),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=lambda x, edge_index: (x[torch.randperm(x.size(0))], edge_index),
        ).to(device)

    def train_model(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(200):  # Typical DGI training can take more epochs
            optimizer.zero_grad()
            pos_z, neg_z, summary = model(self.features, self.edge_index)
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def generate_embeddings(self, model):
        model.eval()
        with torch.no_grad():
            z, _, _ = model(self.features, self.edge_index)
        embeddings = {node: z[i].cpu().numpy() for i, node in enumerate(range(z.size(0)))}
        return embeddings
