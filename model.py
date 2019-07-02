import torch
import torch.nn as nn


######################################################################
# The Encoder
# -----------
#
#
class Encoder(nn.Module):
    def __init__(self, n_words, dim_word, dim_thought):
        super(Encoder, self).__init__()

        self.n_words = n_words
        self.dim_word = dim_word
        self.dim_thought = dim_thought

        self.embedding = nn.Embedding(self.n_words, self.dim_word)
        self.lstm = nn.LSTM(self.dim_word, self.dim_thought)

    def _reverse_embedded(self, x):
        ids = [i for i in range(x.size(0) - 1, -1, -1)]
        ids = torch.tensor(ids, dtype=torch.long, device=x.device)

        rev = x.index_select(0, ids)
        return rev

    def forward(self, sentence):  # sentences: (batch_size, max_len)
        sentence = sentence.transpose(0, 1)  # (max_len, batch_size)
        embedded = self.embedding(sentence)  # (max_len, batch_size, dim_word)
        # embedded = torch.tanh(embeddings)

        # 反転させるとよいらしい
        rev = self._reverse_embedded(embedded)

        output, (hidden, cell) = self.lstm(rev)
        thoughts = hidden[-1]  # (batch_size, dim_thought)

        return embedded, thoughts


######################################################################
# The Decoder
# -----------
#
#
class Decoder(nn.Module):
    def __init__(self, n_words, dim_word, dim_thought, max_len):
        super(Decoder, self).__init__()

        self.n_words = n_words
        self.dim_word = dim_word
        self.dim_thought = dim_thought
        self.max_len = max_len

        self.prev_lstm = nn.LSTM(self.dim_thought + self.dim_word, self.dim_word)
        self.next_lstm = nn.LSTM(self.dim_thought + self.dim_word, self.dim_word)
        self.to_word = nn.Linear(self.dim_word, self.n_words)

    def forward(self, embedded, thoughts):
        thoughts = thoughts.repeat(self.max_len, 1, 1)  # (max_len, batch_size, dim_thought)

        prev_thoughts = thoughts[:, :-1, :]  # (max_len, batch_size-1, dim_thought)
        next_thoughts = thoughts[:, 1:, :]  # (max_len, batch_size-1, dim_thought)

        prev_embedded = embedded[:, :-1, :]  # (max_len, batch_size-1, dim_word)
        next_embedded = embedded[:, 1:, :]  # (max_len, batch_size-1, dim_word)

        delayed_prev_embedded = torch.cat([0 * prev_embedded[-1:, :, :], prev_embedded[:-1, :, :]])  # (max_len, batch_size-1, dim_word)
        delayed_next_embedded = torch.cat([0 * next_embedded[-1:, :, :], next_embedded[:-1, :, :]])  # (max_len, batch_size-1, dim_word)

        prev_pred_embedded, (hidden, cell) = self.prev_lstm(torch.cat([next_thoughts, delayed_prev_embedded], dim=2))  # (max_len, batch_size-1, dim_thought)
        next_pred_embedded, (hidden, cell) = self.next_lstm(torch.cat([prev_thoughts, delayed_next_embedded], dim=2))  # (max_len, batch_size-1, dim_thought)

        # predict actual word
        a, b, c = prev_pred_embedded.size()
        prev_pred = self.to_word(prev_pred_embedded.view(a * b, c)).view(a, b, -1)  # (max_len, batch_size-1, n_words)
        a, b, c = next_pred_embedded.size()
        next_pred = self.to_word(next_pred_embedded.view(a * b, c)).view(a, b, -1)  # (max_len, batch_size-1, n_words)

        # prev_pred = prev_pred.transpose(0, 1).contiguous()
        # next_pred = next_pred.transpose(0, 1).contiguous()
        prev_pred = prev_pred.transpose(0, 1)  # (batch_size-1, max_len, n_words)
        next_pred = next_pred.transpose(0, 1)  # (batch_size-1, max_len, n_words)

        return prev_pred, next_pred


class UniSkip(nn.Module):
    def __init__(self, n_words, dim_word, dim_thought, max_len):
        super(UniSkip, self).__init__()
        self.encoder = Encoder(n_words, dim_word, dim_thought)
        self.decoder = Decoder(n_words, dim_word, dim_thought, max_len)

    def _create_mask(self, x, length):
        mask = x.clone().fill_(0)
        for i, l in enumerate(length):
            for j in range(l):
                mask[i, j] = 1

        return mask

    def forward(self, sentence, length):
        # sentence: (batch_size, max_len)
        # length:   (batch_size)

        # Calculate the skip thought vector
        embedded, thoughts = self.encoder(sentence)

        prev_pred, next_pred = self.decoder(embedded, thoughts)

        prev_mask = self._create_mask(prev_pred, length[:-1])
        next_mask = self._create_mask(next_pred, length[:-1])

        masked_prev_pred = prev_pred * prev_mask
        masked_next_pred = next_pred * next_mask

        return masked_prev_pred, masked_next_pred
