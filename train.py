import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UncertainObjectDataset, collate_fn, collate_fn2
from loss_fn import iej_loss
from model import IEJModel
from utils import init_weights

# from model import SimpleIEJ as IEJModel

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_objects', type=int, default=500)
    parser.add_argument('--epochs_per_turn', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_turns', type=int, default=10)
    parser.add_argument('--dims', type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10], nargs='+')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    num_objects = args.num_objects
    num_epochs = args.epochs_per_turn * args.num_turns
    batch_size = args.batch_size

    device = torch.device(args.device)

    Path('./ckpt').mkdir(parents=True, exist_ok=True)

    for _dim in args.dims:

        # dim = int(2 ** _dim)
        dim = _dim

        train_ds = UncertainObjectDataset(num_objects, dim)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)

        eval_ds = UncertainObjectDataset(100, dim)
        eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn2, num_workers=8)

        model = IEJModel(dim, 4, args.hidden_size, args.num_layers)
        model.apply(init_weights)

        model.to(device)
        model.train()

        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: args.gamma ** (x % args.epochs_per_turn))

        progress_bar = tqdm(total=num_epochs * (len(train_dl) + len(eval_dl)))

        best_acc = -np.inf
        for epoch in range(num_epochs):

            if epoch % args.epochs_per_turn == 0:
                train_ds = UncertainObjectDataset(num_objects, dim)
                train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                      num_workers=8)

            model.train()
            acm_loss = 0.0
            for a, b, epsilon, min_distance in train_dl:
                a, b, epsilon, min_distance = a.to(device), b.to(device), epsilon.to(device), min_distance.to(device)
                optim.zero_grad()
                w = model(a, b)
                loss = iej_loss(w, a, b, epsilon, min_distance)
                loss.backward()
                optim.step()
                acm_loss += loss.item()
                progress_bar.update(1)

            logging.info(f'Epoch {epoch + 1} loss: {acm_loss / len(train_dl)}')

            scheduler.step()

            model.eval()
            with torch.no_grad():
                y_pred = []
                y_true = []
                for a_tensor, b_tensor, epsilons, a_objects, b_objects in eval_dl:
                    a_tensor, b_tensor = a_tensor.to(device), b_tensor.to(device)
                    ws = model(a_tensor, b_tensor).cpu().numpy()
                    for a, b, epsilon, w in zip(a_objects, b_objects, epsilons, ws):
                        delta = w * epsilon * 0.5
                        y_pred.append(int(a.iej(b, epsilon, delta)))
                        y_true.append(int(a.ej(b, epsilon)))
                    progress_bar.update(1)
                acc = np.mean(np.array(y_pred) == np.array(y_true))

            logging.info(f'Epoch {epoch + 1} eval acc: {acc}')

            if acc > best_acc:
                # save model
                torch.save(model.state_dict(), f'./ckpt/iej_{dim}_best.pth')
                best_loss = acc

            torch.save(model.state_dict(), f'./ckpt/iej_{dim}_last.pth')
