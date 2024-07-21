import torch
import torch.nn.functional as F

# from torch_sparse import SparseTensor
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.nn import GCNConv
from tqdm import tqdm


class GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(GCNEncoder, self).__init__()
        self.attention = torch.nn.MultiheadAttention(input_dim, num_heads)

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, return_attention_weights=False):
        x = x.to_dense()

        # Add batch_size dimension
        x = x.unsqueeze(1)

        # Apply Multihead Attention
        x, attn_weights = self.attention(x, x, x)

        # Squeeze to remove the batch_size dimension
        x = x.squeeze(1)

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)

        if return_attention_weights:
            return x, attn_weights
        else:
            return x


class GCNVAE(torch.nn.Module):

    def __init__(self, input_dim, out_dim):
        super(GCNVAE, self).__init__()
        self.enc = GCNEncoder(input_dim, 2 * out_dim)
        self.relu = torch.nn.ReLU()
        self.fc_mu = torch.nn.Linear(2 * out_dim, out_dim)
        self.fc_logvar = torch.nn.Linear(2 * out_dim, out_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index, return_mu_std=False):
        h = self.enc(x, edge_index)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar if return_mu_std else z


if __name__ == "__main__":
    input_dim = 64
    out_dim = 32
    num_nodes = 100
    num_edges = 300

    x = torch.randn(num_nodes, input_dim)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge indices

    model = GCNVAE(input_dim, out_dim)

    model.train()

    z, mu, logvar = model(x, edge_index, return_mu_std=True)

    # Print shapes of outputs (optional)
    print("Output shapes:")
    print("z:", z.shape)
    print("mu:", mu.shape)
    print("logvar:", logvar.shape)

    # Alternatively, if you only want z:
    # z = model(x, edge_index)

    # Print z (optional)
    print("z:", z)
