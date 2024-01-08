import logging
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset import UncertainObjectDataset
# from model import IEJModel
from model import SimpleIEJ as IEJModel
from utils import generate_objects

time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(f'log_{time_string}.txt', encoding='utf-8')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_objects', type=int, default=500)
    parser.add_argument('--num_turns', type=int, default=3)
    parser.add_argument('--dims', type=int, default=[1, 2, 3, 4, 5, 6, 7], nargs='+')
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    num_objects = args.num_objects
    num_turns = args.num_turns

    Path('./results').mkdir(parents=True, exist_ok=True)

    for dim in args.dims:

        objects, _ = generate_objects(num_objects, dim, None)
        eval_ds = UncertainObjectDataset(num_objects, dim, [0.1 + 0.025 * i for i in range(30)])

        model = IEJModel(dim, 4, args.hidden_size, args.num_layers)
        model.load_state_dict(torch.load(f'./ckpt/iej_{dim}_best.pth'))
        model.eval()

        ej = [0, 1]
        iej = [0, 1]
        oiej = [0, 1]
        oiej_dl = [0, 1]

        with torch.no_grad():
            for turn in range(num_turns):
                for idx in tqdm(range(len(eval_ds)), desc=f'Dim {dim} Turn {turn}'):
                    a, b, epsilon = eval_ds[idx]
                    w = model(a.mbr_tensor.unsqueeze(0), b.mbr_tensor.unsqueeze(0)).numpy()[0]
                    delta = w * 0.5 * epsilon

                    ej.append(int(a.ej(b, epsilon)))
                    iej.append(int(a.iej(b, epsilon)))
                    oiej.append(int(a.o_iej(b, epsilon)))
                    oiej_dl.append(int(a.iej(b, epsilon, delta)))

        ej = np.array(ej)
        iej = np.array(iej)
        oiej = np.array(oiej)
        oiej_dl = np.array(oiej_dl)

        logger.info(f'Dim {dim}')
        logger.info('IEJ')
        logger.info('\n' + classification_report(ej, iej, digits=4))
        logger.info('O_IEJ')
        logger.info('\n' + classification_report(ej, oiej, digits=4))
        logger.info('O_IEJ (DL)')
        logger.info('\n' + classification_report(ej, oiej_dl, digits=4))

        # save ej, iej, oiej, oiej_dl with numpy array
        np.save(f'./results/ej_{dim}.npy', ej)
        np.save(f'./results/iej_{dim}.npy', iej)
        np.save(f'./results/oiej_{dim}.npy', oiej)
        np.save(f'./results/oiej_dl_{dim}.npy', oiej_dl)
