import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UncertainObjectDataset, collate_fn
from loss_fn import iej_loss
from model import IEJModel

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    num_objects = 800
    num_epochs = 15
    batch_size = 64

    dims = np.array(list(range(1, 8)), dtype=int)
    dims = np.power(2, dims)

    Path('./ckpt').mkdir(parents=True, exist_ok=True)

    for dim in dims:

        train_ds = UncertainObjectDataset(num_objects, dim, [0.05 * i for i in range(1, 11)])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)

        eval_ds = UncertainObjectDataset(100, dim, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

        model = IEJModel(dim, 4)
        model.train()
        optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.5)

        progress_bar = tqdm(total=num_epochs * (len(train_dl) + len(eval_dl)))

        best_loss = np.inf
        for epoch in range(num_epochs):

            train_ds = UncertainObjectDataset(num_objects, dim, [0.05 * i for i in range(1, 11)])
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)

            model.train()
            for a, b, epsilon, min_distance in train_dl:
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
                    w = model(a, b)
                    loss = iej_loss(w, a, b, epsilon, min_distance)
                    acm_loss += loss.item()
                    progress_bar.update(1)

            logging.info(f'Epoch {epoch + 1} eval loss: {acm_loss / len(eval_dl)}')

            if acm_loss < best_loss:
                # save model
                torch.save(model.state_dict(), f'./ckpt/iej_{dim}_best.pth')
                best_loss = acm_loss

        # model.eval()
        # y_pred = []
        # y_truth = []
        # y_iej = []
        # y_oiej = []
        # with torch.no_grad():
        #     for a, b, epsilon in eval_ds:
        #         w = model(a.mbr_tensor.unsqueeze(0), b.mbr_tensor.unsqueeze(0)).numpy()[0]
        #         delta = w * 0.5 * epsilon
        #         y_pred.append(a.iej(b, epsilon, delta))
        #
        #         y_truth.append(a.ej(b, epsilon))
        #         y_iej.append(a.iej(b, epsilon))
        #         y_oiej.append(a.o_iej(b, epsilon))
