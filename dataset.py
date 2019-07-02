import torch
import torch.utils.data
from janome.tokenizer import Tokenizer

from vocab import load_pickle


EOS = 0
UNK = 1


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_path, word_dict_path, n_words=20000, max_len=30):
        print('Reading text file...')
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                sentences = f.readlines()
        except IOError as e:
            print(e)

        self.sentences = sentences
        self.word_dict = load_pickle(word_dict_path)
        self.n_words = n_words
        self.max_len = max_len
        self.tokenizer = Tokenizer()

    def _tokenize(self, text):
        tokenized = self.tokenizer.tokenize(text, wakati=True)
        return tokenized

    def _setence_to_id(self, sentence):
        ids = [
            self.word_dict.get(w) if self.word_dict.get(w, self.n_words + 1) < self.n_words else UNK for w in sentence
            ][:self.max_len - 1]

        # Padding with EOS(0)
        ids += [EOS] * (self.max_len - len(ids))

        return ids

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tokenized = self._tokenize(sentence)
        ids = self._setence_to_id(tokenized)
        ids = torch.tensor(ids, dtype=torch.long)

        length = min(len(tokenized), self.max_len)
        length = torch.tensor(length, dtype=torch.long)

        return ids, length
