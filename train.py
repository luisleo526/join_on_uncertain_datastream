import torch
from torch.utils.data import DataLoader

from dataset import UncertainObjectDataset, collate_fn
from loss_fn import iej_loss
from model import IEJModel

from tqdm import tqdm

if __name__ == '__main__':

    num_objects = 1000
    dim = 3
    batch_size = 64
    num_epochs = 100

    train_ds = UncertainObjectDataset(num_objects, dim, [0.05 * i for i in range(1, 11)])
    eval_ds = UncertainObjectDataset(100, dim, [0.25])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dl = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = IEJModel(dim, 4)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    progress_bar = tqdm(total=num_epochs * (len(train_dl) + len(eval_dl)))

    for epoch in range(num_epochs):

        acm_loss = 0.0

        model.train()
        for a, b, epsilon, min_distance in train_dl:
            optim.zero_grad()
            w = model(a, b)
            loss = iej_loss(w, a, b, epsilon, min_distance)
            loss.backward()
            optim.step()
            acm_loss += loss.item()
            progress_bar.update(1)

        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {acm_loss}')

        acm_loss = 0.0

        model.eval()
        with torch.no_grad():
            for a, b, epsilon, min_distance in eval_dl:
                w = model(a, b)
                loss = iej_loss(w, a, b, epsilon, min_distance)
                acm_loss += loss.item()
                progress_bar.update(1)
        print(f'Eval Loss: {loss.item()}')
