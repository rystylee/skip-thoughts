import torch
import torch.nn as nn


######################################################################
#
# The Encoder
#
######################################################################
class Encoder(nn.Module):
    def __init__(self, n_words, dim_word, dim_hidden):
        super(Encoder, self).__init__()

        self.n_words = n_words
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden

        self.embedding = nn.Embedding(self.n_words, self.dim_word, padding_idx=0)
        self.lstm = nn.LSTM(self.dim_word, self.dim_hidden, batch_first=True)

    def _reverse_embedded(self, x):
        ids = [i for i in range(x.size(1) - 1, -1, -1)]
        ids = torch.tensor(ids, dtype=torch.long, device=x.device)
        rev_embedded = x.index_select(1, ids)
        return rev_embedded

    def forward(self, word_ids):
        """
        Args:
            word_ids: (batch_size, max_seq_len)

        Returns:
            embedded: (batch_size, max_seq_len, dim_word)
            hidden: (num_layers * num_directons, batch_size, dim_hidden)
            cell: (num_layers * num_directons, batch_size, dim_hidden)
        """

        embedded = self.embedding(word_ids)
        rev_embedded = self._reverse_embedded(embedded)
        out_seq, (hidden, cell) = self.lstm(rev_embedded)
        return embedded, (hidden, cell)


######################################################################
#
# The Decoder
#
######################################################################
class Decoder(nn.Module):
    def __init__(self, n_words, dim_word, dim_hidden, max_seq_len):
        super(Decoder, self).__init__()

        self.n_words = n_words
        self.dim_word = dim_word
        self.dim_hidden = dim_hidden
        self.max_seq_len = max_seq_len

        self.lstm = nn.LSTM(self.dim_word, self.dim_hidden, batch_first=True)
        self.linear = nn.Linear(self.dim_hidden, self.n_words)

    def forward(self, embedded, init_hidden, init_cell):
        """
        Args:
            embedded: (batch_size, max_seq_len, dim_word)
            init_hidden: (num_layers * num_directons, batch_size, dim_hidden)
            init_cell: (num_layers * num_directons, batch_size, dim_hidden)

        Returns:
            out: (batch_size, max_seq_len, n_words)
        """

        out_seq, (hidden, cell) = self.lstm(embedded, (init_hidden, init_cell))
        out = self.linear(out_seq)
        return out


######################################################################
#
# The Uniskip
#
######################################################################
class UniSkip(nn.Module):
    def __init__(self, n_words, dim_word, dim_thought, max_seq_len):
        super(UniSkip, self).__init__()

        self.n_words = n_words
        self.dim_word = dim_word
        self.dim_thought = dim_thought
        self.max_seq_len = max_seq_len

        self.encoder = Encoder(self.n_words, self.dim_word, self.dim_thought)
        self.prev_decoder = Decoder(self.n_words, self.dim_word, self.dim_thought, self.max_seq_len)
        self.next_decoder = Decoder(self.n_words, self.dim_word, self.dim_thought, self.max_seq_len)

    def forward(self, word_ids):
        """
        Args:
            word_ids: (batch_size, max_seq_len)

        Returns:
            prev_pred: (batch_size * max_seq_len, n_words)
            next_pred: (batch_size * max_seq_len, n_words)
        """

        # Encode
        embedded, (hidden, cell) = self.encoder(word_ids)

        # Decode previous sentence
        prev_embedded = embedded[:-1]
        delayed_prev_embedded = torch.cat([0 * prev_embedded[:, :1, :], prev_embedded[:, :-1, :]], dim=1)
        prev_pred = self.prev_decoder(delayed_prev_embedded, hidden[:, :-1, :], cell[:, :-1, :])
        prev_pred = prev_pred.view(-1, self.n_words)

        # Decode next sentence
        next_embedded = embedded[1:]
        delayed_next_embedded = torch.cat([0 * next_embedded[:, :1, :], next_embedded[:, :-1, :]], dim=1)
        next_pred = self.next_decoder(delayed_next_embedded, hidden[:, 1:, :], cell[:, 1:, :])
        next_pred = prev_pred.view(-1, self.n_words)

        return prev_pred, next_pred
