import torch
import torch.nn.functional as F
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.nn import GCNConv
from tqdm import tqdm


class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)

        return x
