import torch
import torch.nn.functional as F

from torch import nn


class MTCondLSTM(nn.Module):
    def __init__(self, vocabs, batch_size) -> None:
        super().__init__()
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.hidden_size = 256
        self.num_layers = 2
        self.prefix_len = 5

        self.embeddings = self._set_embeddings(self.vocabs)
        self.input_dim = self._set_input_dim(self.vocabs)

        # inputsize = sum(embedding_dims) + time
        # if emb dim for each cat feature is 3, input_size=7
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(1 + (self.hidden_size * self.prefix_len), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.act_out = nn.Linear(256, self.vocabs["activity"]["size"])
        self.rt_out = nn.Linear(256, 1)

    def _set_embeddings(self, vocabs):
        for feature in vocabs:
            emb = nn.Embedding(
                num_embeddings=vocabs[feature]["size"],
                embedding_dim=vocabs[feature]["emb_dim"],
            )
            vocabs[feature]["embedding_layer"] = emb

        embeddings = nn.ModuleDict(
            {feature: vocabs[feature]["embedding_layer"] for feature in vocabs}
        )
        return embeddings

    @staticmethod
    def _set_input_dim(vocabs):
        # considering three features only in this work (ac, res, time)
        # thus, the initial input_dim is 3-(n_categorical_features)
        # since resource might be numerical or categorical
        input_dim = 3 - len(vocabs)
        for feature in vocabs:
            input_dim += vocabs[feature]["emb_dim"]
        return input_dim

    def _embed(self, e):
        # ToDo: a better way to track/manage each event feature
        embs = None
        for ix, feature in enumerate(self.vocabs):
            values = e[
                :, :, ix
            ].long()  # selecting the categorical attribute; a better way to do this is need
            emb = self.embeddings[feature]

            if embs is None:
                embs = emb(values)
            else:
                embs = torch.cat((embs, emb(values)), dim=2)

        return torch.cat((embs, e[:, :, ix + 1 :]), dim=2)

    def forward(self, x):
        # x[0].shape=(batch_size, prefix_len, 3)
        # where 0=activity, 1=resource, 2=remaining time
        events, condition = x
        events = self._embed(events)
        encoded, _ = self.lstm(events)

        encoded = encoded.flatten(1)
        conditioned = torch.cat((encoded, condition.view(-1, 1)), dim=1)
        out = self.mlp(conditioned)

        next_act = self.act_out(out)
        next_rt = self.rt_out(out)

        return next_act, next_rt
