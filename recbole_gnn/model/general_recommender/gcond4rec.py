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

        self.hidden_dims: list = config["hidden_dims"]
        # self.n_time_buckets: int = config["discretization"]["timestamp"]["bucket"]
        self.n_time_buckets: int = config["num_time_buckets"]
        self.K: int = config["K"]
        self.num_edges: int = config["num_edges"]

        n_embedding_groups = 3
        embedding_dim = self.hidden_dims[0] // n_embedding_groups
        self.user_embedding = nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=embedding_dim
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=embedding_dim
        )
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

    def forward(
        self,
        user_ids,
        item_ids,
        time_buckets,
        edge_index,
        edge_weight=None,
    ):
        user_features = self.user_embedding(user_ids)
        item_features = self.item_embedding(item_ids)
        time_bucket_features = self.time_bucket_embedding(time_buckets)
        x = torch.cat([user_features, item_features, time_bucket_features], dim=-1)

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
        user_ids = interaction[self.USER_ID]
        item_ids = interaction[self.ITEM_ID]
        time_buckets = interaction[self.TIME_FIELD]
        return self.forward(user_ids, item_ids, time_buckets, self.edge_index, self.edge_weight)

    def full_sort_predict(self, interaction):
        return self.predict(interaction)