import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UncertainObjectDataset, collate_fn
from loss_fn import iej_loss
from model import IEJModel

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_objects', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dims', type=int, default=[1, 2, 3, 4, 5, 6, 7], nargs='+')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    num_objects = args.num_objects
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    device = torch.device(args.device)

    Path('./ckpt').mkdir(parents=True, exist_ok=True)

    for _dim in args.dims:

        dim = int(2 ** _dim)

        train_ds = UncertainObjectDataset(num_objects, dim, [0.05 * i for i in range(1, 11)])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)

        eval_ds = UncertainObjectDataset(100, dim, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

        model = IEJModel(dim, 4)
        model.to(device)
        model.train()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)

        progress_bar = tqdm(total=num_epochs * (len(train_dl) + len(eval_dl)))

        best_loss = np.inf
        for epoch in range(num_epochs):

            if epoch % 5 == 0:
                train_ds = UncertainObjectDataset(num_objects, dim, [0.05 * i for i in range(1, 11)])
                train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                      num_workers=8)

            model.train()
            for a, b, epsilon, min_distance in train_dl:
                a, b, epsilon, min_distance = a.to(device), b.to(device), epsilon.to(device), min_distance.to(device)
                optim.zero_grad()
                w = model(a, b)
                loss = iej_loss(w, a, b, epsilon, min_distance)
                loss.backward()
                optim.step()
                progress_bar.update(1)

            scheduler.step()

            model.eval()
            with torch.no_grad():
                acm_loss = 0.0
                for a, b, epsilon, min_distance in eval_dl:
                    a, b, epsilon, min_distance = a.to(device), b.to(device), epsilon.to(device), min_distance.to(
                        device)
                    w = model(a, b)
                    loss = iej_loss(w, a, b, epsilon, min_distance)
                    acm_loss += loss.item()
                    progress_bar.update(1)

            logging.info(f'Epoch {epoch + 1} eval loss: {acm_loss / len(eval_dl)}')

            if acm_loss < best_loss:
                # save model
                torch.save(model.state_dict(), f'./ckpt/iej_{dim}_best.pth')
                best_loss = acm_loss

            torch.save(model.state_dict(), f'./ckpt/iej_{dim}_last.pth')
