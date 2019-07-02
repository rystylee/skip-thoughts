import argparse
import os
import pprint
from tqdm import tqdm

import torch
import torch.optim
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from dataset import TextDataset
from model import UniSkip


parser = argparse.ArgumentParser()
parser.add_argument('--text_path', type=str, default='data/sample.txt')
parser.add_argument('--word_dict_path', type=str, default='data/word_dict.pkl')
parser.add_argument('--word_count_path', type=str, default='data/word_count.pkl')

parser.add_argument('--dim_word', type=int, default=620, help='The dimension of word embeddings')
parser.add_argument('--dim_thought', type=int, default=2400, help='The dimension of skip-thought vector')
parser.add_argument('--n_words', type=int, default=20000, help='')
parser.add_argument('--max_len', type=int, default=30)

parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.0003)

parser.add_argument('--reuse', type=str, default=None, help='The path of model state to load for reuse')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--log_freq', type=int, default=20)
parser.add_argument('--save_dir', type=str, default='result')
parser.add_argument('--save_freq', type=int, default=20)

args = parser.parse_args()
pprint.pprint(args)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    dataset = TextDataset(
        args.text_path,
        args.word_dict_path,
        args.n_words,
        args.max_len
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        )

    model = UniSkip(args.n_words, args.dim_word, args.dim_thought, args.max_len).to(device)
    if args.reuse is not None:
        model.load_state_dict(torch.load(args.reuse))
        print('Loaded the state dict!')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(args.log_dir)

    print('Start training...')
    loss_list = []
    itr = 0
    for epoch in range(args.n_epoch):
        print('Start epoch: [{}]'.format(epoch))
        for i, (ids, length) in tqdm(enumerate(dataloader), desc='', total=len(dataloader)):
            ids = ids.to(device)
            length = length.to(device)

            prev_pred, next_pred = model(ids, length)

            # losses
            prev_loss = F.cross_entropy(prev_pred.view(-1, args.n_words), ids[:-1, :].view(-1))
            next_loss = F.cross_entropy(next_pred.view(-1, args.n_words), ids[1:, :].view(-1))
            loss = prev_loss + next_loss
            loss_list.append(loss.item())

            if itr % args.log_freq == 0:
                writer.add_scalar('loss', loss.item(), itr)

            if itr % args.save_freq == 0:
                model_path = '{}/{}_itr.pth'.format(args.save_dir, itr)
                torch.save(model.state_dict(), model_path)
                print('Saved model to {}...'.format(model_path))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            itr += 1

    writer.close()
