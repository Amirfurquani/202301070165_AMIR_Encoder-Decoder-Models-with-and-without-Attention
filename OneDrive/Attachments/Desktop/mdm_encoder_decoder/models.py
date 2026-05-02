"""
Encoder-Decoder Models With and Without Attention
==================================================
Part 3: Implementation - Model Architectures
Paper Reference: "Efficient Machine Translation with BiLSTM-Attention" (2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ============================================================
# MODEL 1: Seq2Seq WITHOUT Attention (Baseline)
# ============================================================

class EncoderNoAttn(nn.Module):
    """Simple LSTM Encoder - encodes source sequence into a fixed context vector."""
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        # Only return final hidden state as fixed context vector
        return hidden, cell


class DecoderNoAttn(nn.Module):
    """Simple LSTM Decoder - uses only the final encoder hidden state."""
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, cell):
        # input_tok: (batch,)
        input_tok = input_tok.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input_tok))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2SeqNoAttention(nn.Module):
    """Complete Seq2Seq model WITHOUT attention mechanism."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input_tok = trg[:, 0]  # <sos> token

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input_tok = trg[:, t] if random.random() < teacher_forcing_ratio else top1

        return outputs


# ============================================================
# MODEL 2: Seq2Seq WITH Bahdanau Attention
# ============================================================

class EncoderWithAttn(nn.Module):
    """Bidirectional LSTM Encoder for attention-based model."""
    def __init__(self, input_dim, emb_dim, enc_hidden, dec_hidden, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, enc_hidden, n_layers, dropout=dropout,
                           batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(enc_hidden * 2, dec_hidden)
        self.fc_cell = nn.Linear(enc_hidden * 2, dec_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        enc_outputs, (hidden, cell) = self.lstm(embedded)
        # hidden: (n_layers*2, batch, enc_hidden) -> (n_layers, batch, dec_hidden)
        # Concatenate forward and backward for each layer
        n_layers = hidden.shape[0] // 2
        hidden_fwd = hidden[0::2]  # (n_layers, batch, enc_hidden)
        hidden_bwd = hidden[1::2]
        hidden = torch.tanh(self.fc_hidden(torch.cat([hidden_fwd, hidden_bwd], dim=2)))
        cell_fwd = cell[0::2]
        cell_bwd = cell[1::2]
        cell = torch.tanh(self.fc_cell(torch.cat([cell_fwd, cell_bwd], dim=2)))
        return enc_outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention Mechanism.
    
    Computes alignment scores using a learned additive function:
        score(s_t, h_j) = v^T * tanh(W_s * s_t + W_h * h_j)
    """
    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.W_s = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.W_h = nn.Linear(enc_hidden * 2, dec_hidden, bias=False)
        self.v = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, dec_hidden)
        # encoder_outputs: (batch, src_len, enc_hidden*2)
        src_len = encoder_outputs.shape[1]

        hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.W_s(hidden_expanded) + self.W_h(encoder_outputs))
        attention_scores = self.v(energy).squeeze(2)  # (batch, src_len)
        attention_weights = F.softmax(attention_scores, dim=1)
        return attention_weights


class DecoderWithAttn(nn.Module):
    """LSTM Decoder with Bahdanau Attention."""
    def __init__(self, output_dim, emb_dim, enc_hidden, dec_hidden, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = BahdanauAttention(enc_hidden, dec_hidden)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim + enc_hidden * 2, dec_hidden, n_layers,
                           dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(dec_hidden + enc_hidden * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, cell, encoder_outputs):
        input_tok = input_tok.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_tok))  # (batch,1,emb)

        attn_weights = self.attention(hidden[-1], encoder_outputs)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch,1,enc*2)

        lstm_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.fc_out(torch.cat([output.squeeze(1), context.squeeze(1),
                                            embedded.squeeze(1)], dim=1))
        return prediction, hidden, cell, attn_weights


class Seq2SeqWithAttention(nn.Module):
    """Complete Seq2Seq model WITH Bahdanau Attention mechanism."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        input_tok = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell, attn_w = self.decoder(input_tok, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            attentions[:, t, :] = attn_w
            top1 = output.argmax(1)
            input_tok = trg[:, t] if random.random() < teacher_forcing_ratio else top1

        return outputs, attentions
