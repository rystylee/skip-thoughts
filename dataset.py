import torch
import torch.utils.data
from janome.tokenizer import Tokenizer

from vocab import load_pickle, preprocess


EOS = 0
UNK = 1


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_path, word_dict_path, n_words=20000, max_seq_len=30, encoding='shift-jis'):
        print('Reading text file...')
        try:
            with open(text_path, 'r', encoding=encoding) as f:
                sentences = f.read()
        except IOError as e:
            print(e)

        self.sentences = preprocess(sentences)
        self.word2id = load_pickle(word_dict_path)
        self.id2word = {v: k for k, v, in self.word2id.items()}
        self.n_words = n_words
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer()

        print('The num of sentences is {}'.format(len(self.sentences)))

    def _tokenize(self, text):
        tokenized = self.tokenizer.tokenize(text, wakati=True)
        return tokenized

    def _setence_to_id(self, sentence):
        ids = [
            self.word2id.get(w) if self.word2id.get(w, self.n_words + 1) < self.n_words else UNK for w in sentence
            ][:self.max_seq_len - 1]

        # Padding with EOS(0)
        ids += [EOS] * (self.max_seq_len - len(ids))

        return ids

    def _id_to_setence(self, ids):
        sentence = [self.id2word.get(i) for i in ids]
        return sentence

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokenized = self._tokenize(sentence)
        word_ids = self._setence_to_id(tokenized)
        word_ids = torch.tensor(word_ids, dtype=torch.long)

        return word_ids
