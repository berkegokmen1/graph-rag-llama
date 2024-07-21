import torch
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
from models.base import BaseGraphEmbeddingModel


class Node2VecModel(BaseGraphEmbeddingModel):
    def process_graph(self, graph, device):
        self.edge_index = torch.tensor(list(graph.edges())).t().contiguous().to(device)
        self.graph = graph

    def create_model(self, device):
        return Node2Vec(
            self.edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=10,
            context_size=5,
            walks_per_node=50,
            num_negative_samples=1,
            p=1,
            q=1,
            sparse=True,
        ).to(device)

    def train_model(self, model, device):
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.0001)

        pbar = tqdm(range(10000), desc="Training")

        model.train()
        for epoch in pbar:
            total_loss = 0
            for pos_rw, neg_rw in tqdm(loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            pbar.set_description(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")
            # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader)}")

    def generate_embeddings(self, model, device):
        model.eval()
        embeddings = {}
        with torch.no_grad():
            for i, node in enumerate(self.graph.nodes()):
                embeddings[node] = model(torch.tensor([i]).to(device)).cpu().numpy()[0]
        return embeddings
