import torch

from graph_condensation.parametrized_adj import PGE
from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender


class GCond:
    r"""[ICLR'22] [KDD'22] [IJCAI'24] Implementation of 'Graph Condensation for Graph Neural Networks'"""

    def __init__(self, config, dataset, backbone: GeneralGraphRecommender):
        # load dataset info
        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        self.backbone = backbone

        self.device = config["device"]
        self.reduction_rate = config.get("reduction_rate", 0.5)

        self.n_users_syn = int(self.n_users * self.reduction_rate)
        self.n_items_syn = int(self.n_items * self.reduction_rate)
        self.latent_dim = config["embedding_size"]
        self.user_syn_param = torch.nn.Parameter(
            torch.FloatTensor(self.n_users_syn, self.latent_dim)
        ).to(self.device)
        self.item_syn_param = torch.nn.Parameter(
            torch.FloatTensor(self.n_items_syn, self.latent_dim)
        ).to(self.device)

        self.pge = PGE(
            config, self.latent_dim, self.n_users + self.n_items, nhid=128, nlayers=3
        )  # TODO: refine the parameters

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: try different initialization methods
        #  self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
        torch.nn.init.xavier_uniform_(self.user_syn_param)
        torch.nn.init.xavier_uniform_(self.item_syn_param)

    def fit(self, verbose=True): ...
