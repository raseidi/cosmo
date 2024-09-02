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
