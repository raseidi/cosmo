import torch
from torch import nn
from torch.nn import functional as F


class ConstrainedRNNCell(nn.Module):

    """The motivation under this model it to make
    the RNN cell aware of the constraints. The idea
    is to add a new term to the hidden state update
    equation that is a linear combination of the
    constraint vector and the hidden state. Thus,
    the model will be able to learn both: (i) to use
    the constraints to update the hidden state and
    (ii) the recurrence aspect of the given sequence.

    At each step, the goal is to teach the model to
    learn the relation between the current event and
    the previous hidden state based on the constraints."""

    def __init__(self, input_size, constraint_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.constraint_size = constraint_size

        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W_ch = nn.Parameter(torch.randn(constraint_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        self.adjust_input = nn.Linear(input_size, hidden_size, bias=False)

        self.init_params()

    def forward(self, x, c, h):
        c = c.float().squeeze()
        q = x @ self.W_xh
        w = h @ self.W_hh
        e = c @ self.W_ch

        # h = torch.tanh(q + w + e + self.b_h)
        # relu as activation function
        h = F.gelu(q + w + e + self.b_h)
        # h = F.gelu(q + w + self.b_h) # unconstrained
        x = self.adjust_input(x)
        return h + x

    def init_params(self):
        stdv = 1.0 / (self.hidden_size ** (1 / 2))
        for weight in self.parameters():
            weight.data.normal_(-stdv, stdv)

        # remove bias from forget gate
        # self.W_hh.data[:, self.hidden_size // 3 : self.hidden_size // 2] = 0.0


# class CRNN(nn.Module):
#     def __init__(
#         self, vocabs, continuous_size, constraint_size, input_size, hidden_size, n_layers=1, batch_first=True
#     ):
#         super(CRNN, self).__init__()
#         self.vocabs = vocabs
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.batch_first = batch_first
#         self.continuous_size = continuous_size
#         self.vocab_dims = [len(vocab) for vocab in vocabs.values()]
#         self.n_features = len(self.vocab_dims) + continuous_size # n vocabularies + continuous features

#         self.input_cont = nn.Linear(continuous_size, input_size // self.n_features)
#         self.input_cat = nn.ModuleList(
#             [
#                 nn.Embedding(
#                     dim, input_size // self.n_features, padding_idx=0
#                 )
#                 for dim in self.vocab_dims
#             ]
#         )

#         # backbone
#         self.rnn = [ConstrainedRNN(
#             input_size=input_size, #+continuous_size,
#             constraint_size=constraint_size,
#             hidden_size=hidden_size,
#         )]
#         if n_layers > 1:
#             for _ in range(n_layers - 1):
#                 self.rnn.append(ConstrainedRNN(
#                     input_size=hidden_size,
#                     constraint_size=constraint_size,
#                     hidden_size=hidden_size,
#                 ))
#         self.rnn = nn.ModuleList(self.rnn)


#         self.batch_norm_rnn = nn.BatchNorm1d(hidden_size)
#         self.linear = nn.Linear(hidden_size, hidden_size)

#         # multi-output layers
#         self.classifier = nn.ModuleList(
#             [nn.Linear(hidden_size, dim) for dim in self.vocab_dims]
#         )
#         self.regressor = nn.Linear(hidden_size, 1)

#         self.init_params()

#     def forward(self, x: tuple[torch.tensor, torch.tensor], constraints, h=None):
#         cat, num = x
#         lengths = [sum(el != 0).item() for el in cat]

#         x = torch.cat(
#             [
#                 self.input_cont(num),
#                 # num,
#                 torch.cat(
#                     [self.input_cat[i](cat[..., i]) for i in range(cat.shape[-1])],
#                     dim=-1,
#                 ),
#             ],
#             dim=-1,
#         )
#         # x = torch.cat(
#         #     [self.input_cat[i](cat[..., i]) for i in range(cat.shape[-1])],
#         #     dim=-1,
#         # )

#         # it has shape (n_layers, batch, hidden_size)
#         if h is None:
#             batch_size = x.size(0) if self.batch_first else x.size(1)
#             h = torch.zeros(len(self.rnn), batch_size, self.hidden_size, device=x.device)
#         hiddens = []
#         for l,h_t in zip(self.rnn, h):
#             x, h = l(x=x, constraints=constraints, lengths=lengths, h_t=h_t)
#             hiddens.append(h)
#         # out, hidden = self.rnn(x, c, lengths)
#         out = x
#         hiddens = torch.stack(hiddens)
#         out = self.batch_norm_rnn(out.transpose(1, 2)).transpose(1, 2)
#         out = F.relu(self.linear(out))

#         cat_out = torch.cat(
#             [self.classifier[i](out) for i in range(len(self.vocabs))],
#             dim=-1,
#         )
#         time_out = self.regressor(out)
#         return cat_out, time_out, hiddens

#     def init_params(self):
#         for weight in self.parameters():
#             if len(weight.shape) > 1:
#                 # nn.init.xavier_uniform_(weight)
#                 # nn.init.orthogonal_(weight)
#                 # nn.init.normal_(weight, mean=0, std=0.01)
#                 nn.init.xavier_normal_(weight)
