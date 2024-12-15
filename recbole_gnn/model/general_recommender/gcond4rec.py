import torch
from recbole.model.init import xavier_uniform_initialization
from recbole.utils import InputType
from torch import nn
from torch_geometric.nn import SGConv, MessagePassing

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender


class GCond4Rec(GeneralGraphRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.TIME_FIELD = config["TIME_FIELD"]  # i.e., dataset.time_field, e.g., 'timestamp'
        self.RATING_FIELD = config["RATING_FIELD"]  # e.g., 'rating'
        self.INDEX_FIELD = config["INDEX_FIELD"]  # e.g., 'idx'

        self.hidden_dims: list = config["hidden_dims"]
        # self.n_time_buckets: int = config["discretization"]["timestamp"]["bucket"]
        self.n_time_buckets: int = config["num_time_buckets"]
        self.K: int = config["K"]
        self.num_edges: int = config["num_edges"]

        n_embedding_groups = 3
        embedding_dim = self.hidden_dims[0] // n_embedding_groups
        self.user_embedding = nn.Embedding(num_embeddings=self.n_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.n_items, embedding_dim=embedding_dim)
        self.time_bucket_embedding = nn.Embedding(
            num_embeddings=self.n_time_buckets,
            embedding_dim=self.hidden_dims[0] - 2 * embedding_dim,
        )
        self.edge_index, self.edge_weight = dataset.get_random_edge_index(self.num_edges)

        layers = []
        input_dim = self.hidden_dims[0]
        for output_dim in self.hidden_dims[1:]:
            layers.append(SGConv(input_dim, output_dim, K=self.K, add_self_loops=False))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(output_dim))
            input_dim = output_dim

        layers.append(nn.Linear(self.hidden_dims[-1], 1))
        layers.append(nn.Flatten(start_dim=0))
        self.layers = nn.ModuleList(layers)

        self.loss = nn.MSELoss()
        self.apply(xavier_uniform_initialization)

    def get_batch_edges(self, edge_index, edge_weight, batch_indices):
        batch_indices = batch_indices.cpu()
        # Create a mask to select edges where both nodes are in the batch
        mask = torch.isin(edge_index[0], batch_indices) & torch.isin(edge_index[1], batch_indices)

        # Select the edges and edge weights for the batch
        edge_index_batch = edge_index[:, mask]
        edge_weight_batch = edge_weight[mask]

        # Map the original indices to batch indices
        index_map = {idx.item(): i for i, idx in enumerate(batch_indices)}
        edge_index_batch = torch.tensor([[index_map[idx.item()] for idx in edge_index_batch[0]],
                                         [index_map[idx.item()] for idx in edge_index_batch[1]]], dtype=torch.long)

        edge_index_batch = edge_index_batch.to(self.device)
        edge_weight_batch = edge_weight_batch.to(self.device)
        return edge_index_batch, edge_weight_batch

    def forward(
        self,
        indices,
        user_ids,
        item_ids,
        time_buckets,
    ):
        user_features = self.user_embedding(user_ids)
        item_features = self.item_embedding(item_ids)
        time_bucket_features = self.time_bucket_embedding(time_buckets)
        x = torch.cat([user_features, item_features, time_bucket_features], dim=-1)

        edge_index, edge_weight = self.get_batch_edges(self.edge_index, self.edge_weight, indices)

        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x

    def calculate_loss(self, interaction):
        ratings = interaction[self.RATING_FIELD]
        predictions = self.predict(interaction)

        return self.loss(predictions, ratings)

    def predict(self, interaction):
        indices = interaction[self.INDEX_FIELD]
        user_ids = interaction[self.USER_ID]
        item_ids = interaction[self.ITEM_ID]
        time_buckets = interaction[self.TIME_FIELD]
        return self.forward(indices, user_ids, item_ids, time_buckets)

    def full_sort_predict(self, interaction):
        # TODO: Implement full_sort_predict
        user = interaction[self.USER_ID] # This is the only info we can use
        ...
