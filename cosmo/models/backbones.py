import torch
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from .crnn import ConstrainedRNNCell

from peft import LoraModel, LoraConfig, TaskType


class GPT2(nn.Module):
    def __init__(self, lora: bool = False, r_rank=4, lora_alpha=16):
        super(GPT2, self).__init__()
        self.backbone = GPT2Model.from_pretrained("gpt2")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Adjust based on your specific task
            # inference_mode=False,
            target_modules=["c_attn"],  # , "c_proj", ],
            r=r_rank,
            lora_alpha=lora_alpha,
            # lora_dropout=0.1
        )

        if lora:
            self.backbone = LoraModel(self.backbone, peft_config, "default")
        else:
            for name, param in self.backbone.named_parameters():
                # freeze all parameters except the layernorm and positional embeddings
                if "ln" in name or "wpe" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x, attention_mask):
        output_shape = x.shape
        past_length = 0
        past_key_values = tuple([None] * len(self.backbone.h))

        attention_mask = attention_mask.view(x.shape[0], -1)
        attention_mask = attention_mask.long()
        attention_mask = attention_mask[:, None, None, :]
        # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.backbone.dtype).min
        for i, (block, layer_past) in enumerate(zip(self.backbone.h, past_key_values)):
            outputs = block(
                x,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=True,
                output_attentions=False,
            )
            x = outputs[0]

        hidden_states = self.backbone.ln_f(x)
        hidden_states = hidden_states.view(output_shape)

        return hidden_states


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


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        max_seq_len=512,
    ):
        super(VanillaTransformer, self).__init__()

        # self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.input_size = input_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(max_seq_len, input_size).unsqueeze(0), requires_grad=True
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, x, src_mask, src_key_padding_mask):
        seq_len = x.size(1)
        # emb = self.embedding(x)
        src_key_padding_mask = src_key_padding_mask.view(x.shape[0], -1)
        src_key_padding_mask = src_key_padding_mask.long()
        # attention_mask = attention_mask[:, None, None, :]
        # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        src_key_padding_mask = (
            1.0 - src_key_padding_mask
        )  # * torch.finfo(self.transformer_encoder.dtype).min

        x += self.positional_encoding[:, :seq_len, :]
        out = self.transformer_encoder(
            x, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        return out


class VanillaMHA(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        n_heads,
        dropout=0.1,
        max_seq_len=512,
    ):
        super(VanillaMHA, self).__init__()
        self.input_size = input_size
        self.n_heads = n_heads
        self.output_size = output_size
        self.dropout = dropout

        self.pos_emb = nn.Parameter(
            torch.randn(max_seq_len, input_size).unsqueeze(0), requires_grad=True
        )
        self.norm = nn.LayerNorm(input_size)
        # self.to_qkv = nn.Linear(input_size, input_size * n_heads * 3, bias=False)
        self.mha = nn.MultiheadAttention(
            input_size, n_heads, dropout=dropout, batch_first=True
        )
        self.to_out = nn.Linear(
            input_size, output_size
        )  # no need to multiply by n_heads cause pytorch already does that

    def forward(self, x, src_key_padding_mask, src_mask=None):
        x = self.norm(x)
        x += self.pos_emb[:, : x.size(1), :]
        # q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), (q, k, v))

        src_key_padding_mask = src_key_padding_mask.view(x.shape[0], -1)
        src_key_padding_mask = src_key_padding_mask.long()
        src_key_padding_mask = 1.0 - src_key_padding_mask

        attn_output, _ = self.mha(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
        )

        return self.to_out(x + attn_output)
