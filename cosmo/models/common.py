import torch
from torch import nn
from torch.nn import functional as F


class InLayer(nn.Module):
    def __init__(
        self,
        vocabs: list,
        n_continuous: int,
        n_constraints: int,
        embedding_size: int = 768,
    ):

        super(InLayer, self).__init__()
        self.embedding_size = embedding_size

        self.cat_dims = [len(vocab) for vocab in vocabs.values()]
        self.n_features = len(vocabs) + n_continuous
        self.n_continuous = n_continuous

        self.cat_layer = nn.ModuleList(
            [
                nn.Embedding(dim, embedding_size // self.n_features, padding_idx=0)
                for dim in self.cat_dims
            ]
        )

        if self.n_continuous > 0:
            self.cont_layer = nn.Linear(n_continuous, embedding_size // self.n_features)

        self.init_params()

    def init_params(self):
        for layer in self.cat_layer:
            nn.init.xavier_uniform_(layer.weight)

        if self.n_continuous > 0:
            nn.init.xavier_uniform_(self.cont_layer.weight)

    def forward(self, x, constraints):
        cat, num = x
        lengths = [sum(el != 0).item() for el in cat]

        if constraints.dim() == 3:
            constraints = constraints.squeeze(1)
        x = torch.cat(
            [self.cat_layer[i](cat[..., i]) for i in range(cat.shape[-1])],
            dim=-1,
        )

        if self.n_continuous > 0:
            x = torch.cat(
                [x, self.cont_layer(num)],
                dim=-1,
            )
        # constraints = self.constraint_layer(constraints.float())

        # constraints = constraints.unsqueeze(1).expand(-1, x.shape[1], -1)
        # x = torch.cat([x, constraints], dim=-1)
        return x, lengths


class OutLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(OutLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)
