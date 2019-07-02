#
# ref: https://github.com/sanyam5/skip-thoughts
# ref: https://github.com/ryankiros/skip-thoughts
#
# Constructing and loading dictionaries
#

import pickle as pkl
from collections import OrderedDict
import argparse
from janome.tokenizer import Tokenizer


def tokenize(text):
    tokenizer = Tokenizer()
    tokenized_text = []
    for cc in text:
        tmp = ''
        for word in tokenizer.tokenize(cc, wakati=True):
            tmp += word
            tmp += ' '
        tokenized_text.append(tmp)
    return tokenized_text


def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    word_count = {}
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in word_count:
                word_count[w] = 0
            word_count[w] += 1

    sorted_words = sorted(list(word_count.keys()), key=lambda x: word_count[x], reverse=True)

    word_dict = OrderedDict()
    for idx, word in enumerate(sorted_words):
        word_dict[word] = idx + 2  # 0: <eos>, 1: <unk>

    return word_dict, word_count


def save_dictionary(word_dict, word_count, path):
    try:
        with open('{}/word_dict.pkl'.format(path), 'wb') as f:
            pkl.dump(word_dict, f)
    except IOError as e:
        print(e)

    try:
        with open('{}/word_count.pkl'.format(path), 'wb') as f:
            pkl.dump(word_dict, f)
    except IOError as e:
        print(e)


def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            data = pkl.load(f)
        return data
    except IOError as e:
        print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--text_file', type=str, default='data/sample.txt')
    args = parser.parse_args()

    print('Reading text file from {}...'.format(args.text_file))
    try:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.readlines()
    except IOError as e:
        print(e)
    print('Done.')

    print('Tokenizing...')
    tokenized_text = tokenize(text)
    print('Done.')

    print('Buiding dictionary..')
    word_dict, word_count = build_dictionary(tokenized_text)
    print('Done.')

    print('Got {} unique words. Saving to file {}'.format(len(word_dict), args.data_dir))
    print('Saving dictionary to {}..'.format(args.data_dir))
    save_dictionary(word_dict, word_count, args.data_dir)
    print('Done.')
