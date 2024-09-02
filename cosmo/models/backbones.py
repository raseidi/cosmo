import torch
from torch import nn

from .crnn import ConstrainedRNNCell


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, architecture="rnn"):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        if architecture == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
        elif architecture == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)

    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        # detach
        if isinstance(h, tuple):
            h = tuple([h_.detach() for h_ in h])
        else:
            h = h.detach()

        return out, h


class ConstrainedRNN(nn.Module):
    def __init__(self, input_size, n_constraints, hidden_size, batch_first=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_constraints = n_constraints
        self.input_size = input_size
        self.batch_first = batch_first

        self.rnn_cell = ConstrainedRNNCell(input_size, n_constraints, hidden_size)

    def forward(self, x: torch.Tensor, constraints, lengths, h_t=None):
        # Initialize the hidden state to zeros
        # x is of shape (seq_len, batch, input_size)
        # constraints is of shape (batch, n_constraints)

        # List to store outputs at each time step

        if self.batch_first:
            x = x.transpose(0, 1)
        if h_t is None:
            h_t = torch.zeros(x.size(1), self.hidden_size, device=x.device)

        # recurrence loop
        # for i in range(x.size(0)):  # x.size(0) is seq_len
        #     h_t = self.rnn_cell(x[i], c, h_t)
        #     outputs.append(h_t)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=False, enforce_sorted=False
        )

        packed_data, batch_sizes = packed.data, packed.batch_sizes
        c = constraints[packed.sorted_indices]
        outputs = []
        data_index = 0
        for batch_size in batch_sizes:
            # Get the input for this time step
            timestep_input = packed_data[data_index : data_index + batch_size, :]
            if h_t.size(0) > batch_size:
                # this is necessary when the last batch is smaller
                h_t = h_t[:batch_size, :]

            h_t = self.rnn_cell(timestep_input, c[:batch_size], h_t)
            outputs.append(h_t)
            # Update the data index
            data_index += batch_size

        # Stack outputs (seq_len, batch, hidden_size)
        outputs = torch.cat(outputs)
        repacked_output = nn.utils.rnn.PackedSequence(
            outputs, batch_sizes, unsorted_indices=packed.unsorted_indices
        )
        unpacked_output, lengths = nn.utils.rnn.pad_packed_sequence(
            repacked_output, batch_first=False
        )
        unpacked_output = unpacked_output.transpose(0, 1)
        # my checking assumes it is right xD
        # assert torch.all(unpacked_output == torch.stack(outputs))
        lengths = [l - 1 for l in lengths]
        # this return the last state of each sequence

        # shape is (batch, seq_len, hidden_size)
        # for each batch, get the last seq state; lengths is the index of the last state;
        return unpacked_output, unpacked_output[range(len(lengths)), lengths, :]
