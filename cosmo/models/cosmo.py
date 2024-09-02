import torch
from torch import nn
from torch.nn import functional as F
from cosmo.models import InLayer, OutLayer
from cosmo.models import backbones as bb


class Cosmo(nn.Module):
    def __init__(
        self,
        vocabs: list,
        n_continuous: int,
        n_constraints: int = None,
        backbone_model: str = "crnn",
        embedding_size: int = 32,
        hidden_size: int = 128,
        n_layers: int = 1,
        batch_first: bool = True,
    ):
        super(Cosmo, self).__init__()
        if backbone_model != "vanilla":
            assert (
                n_constraints is not None
            ), "`n_constraints` must be specified for non-vanilla backbones"

        self.n_vocabs = vocabs
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.n_continuous = n_continuous
        self.n_constraints = n_constraints
        self.vocabs = vocabs
        self.n_layers = n_layers
        self.backbone_model = backbone_model
        self.batch_first = batch_first

        if backbone_model == "vanilla":
            self.backbone = bb.VanillaRNN(
                input_size=embedding_size,
                hidden_size=hidden_size,
                n_layers=n_layers,
                architecture="rnn",
            )
        elif backbone_model == "crnn":
            self.backbone = [
                bb.ConstrainedRNN(
                    input_size=embedding_size,
                    n_constraints=n_constraints,
                    hidden_size=hidden_size,
                )
            ]
            if n_layers > 1:
                for _ in range(n_layers - 1):
                    self.backbone.append(
                        bb.ConstrainedRNN(
                            input_size=hidden_size,
                            n_constraints=n_constraints,
                            hidden_size=hidden_size,
                        )
                    )
            self.backbone = nn.ModuleList(self.backbone)
        else:
            raise ValueError(f"backbone {backbone_model} not supported")

        self.in_layer = InLayer(
            vocabs=vocabs,
            n_continuous=n_continuous,
            n_constraints=n_constraints,
            embedding_size=self.input_size,  # backbone_kwargs["input_size"],
        )

        self.batch_norm_backbone = nn.BatchNorm1d(self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size // 2)

        self.classifier = nn.ModuleList(
            [
                OutLayer(self.hidden_size // 2, dim)
                for dim in [len(vocab) for vocab in vocabs.values()]
            ]
        )

        if n_continuous > 0:
            self.regressor = OutLayer(self.hidden_size // 2, n_continuous)

        self.init_params()

    def forward(self, x, constraints=None, mask=None, h=None):
        # in layer
        x, lengths = self.in_layer(x, constraints)
        hiddens = []

        # backbone pass
        if self.backbone_model == "vanilla":
            out, _ = self.backbone(x)
        elif self.backbone_model == "crnn":
            if h is None:
                batch_size = x.size(0) if self.batch_first else x.size(1)
                h = torch.zeros(
                    len(self.backbone), batch_size, self.hidden_size, device=x.device
                )

            for layer, h_t in zip(self.backbone, h):
                x, h = layer(x=x, constraints=constraints, lengths=lengths, h_t=h_t)
            hiddens.append(h)
            hiddens = torch.stack(hiddens)
            out = x

        # bn and non-linearity
        out = self.batch_norm_backbone(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(self.linear(out))

        # output layers
        cat_out = torch.cat(
            [self.classifier[i](out) for i in range(len(self.vocabs))],
            dim=-1,
        )

        if self.n_continuous > 0:
            time_out = self.regressor(out)
        else:
            time_out = None
        return cat_out, time_out, out

    def init_params(self):
        for weight in self.parameters():
            if len(weight.shape) > 1:
                # nn.init.xavier_uniform_(weight)
                # nn.init.orthogonal_(weight)
                # nn.init.normal_(weight, mean=0, std=0.01)
                nn.init.xavier_normal_(weight)

