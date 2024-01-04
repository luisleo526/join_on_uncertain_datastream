from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UncertainObjectDataset, collate_fn
from loss_fn import iej_loss
from model import IEJModel

if __name__ == '__main__':

    num_objects = 500
    num_epochs = 10
    batch_size = 64

    dims = np.array(list(range(1, 8)), dtype=int)
    dims = np.power(2, dims)

    Path('./ckpt').mkdir(parents=True, exist_ok=True)

    for dim in dims:

        train_ds = UncertainObjectDataset(num_objects, dim, [0.05 * i for i in range(1, 11)])
        eval_ds = UncertainObjectDataset(100, dim, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)

        model = IEJModel(dim, 4)
        model.train()
        optim = torch.optim.Adam(model.parameters(), lr=0.001)

        progress_bar = tqdm(total=num_epochs * (len(train_dl)))

        model.train()
        for epoch in range(num_epochs):

            acm_loss = 0.0

            for a, b, epsilon, min_distance in train_dl:
                optim.zero_grad()
                w = model(a, b)
                loss = iej_loss(w, a, b, epsilon, min_distance)
                loss.backward()
                optim.step()
                acm_loss += loss.item()
                progress_bar.update(1)

            print(f'Epoch {epoch + 1} loss: {acm_loss / len(train_dl)}')

            # save model
            torch.save(model.state_dict(), f'./ckpt/iej_{dim}_{epoch + 1}.pth')

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
