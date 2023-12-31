import logging
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset import UncertainObjectDataset
# from model import IEJModel
from model import SimpleIEJ as IEJModel

time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(f'static_log_{time_string}.txt', encoding='utf-8')
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
    parser.add_argument('--num_objects', type=int, default=100)
    parser.add_argument('--num_turns', type=int, default=1)
    parser.add_argument('--dims', type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10], nargs='+')
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--ckpt_type', choices=['best', 'last'], default='last')
    parser.add_argument('--cmt', type=str, default='precision')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    num_objects = args.num_objects
    num_turns = args.num_turns

    Path('./results').mkdir(parents=True, exist_ok=True)

    results = Namespace(iej=[], oiej=[], oiej_dl=[])

    for dim in args.dims:

        eval_ds = UncertainObjectDataset(num_objects, dim)

        model = IEJModel(dim, 4, args.hidden_size, args.num_layers)
        model.load_state_dict(torch.load(f'./ckpt/iej_{dim}_{args.ckpt_type}.pth', map_location='cpu'))
        model.eval()

        ej = []
        iej = []
        oiej = []
        oiej_dl = []

        with torch.no_grad():
            for turn in range(num_turns):
                for idx in tqdm(range(len(eval_ds)), desc=f'Dim {dim} Turn {turn}'):
                    a, b = eval_ds[idx]
                    w = model(a.mbr_tensor.unsqueeze(0), b.mbr_tensor.unsqueeze(0)).numpy()[0]
                    epsilon = (a % b) * 0.95 if idx % 2 == 0 else (a % b) * 1.05
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

        results.iej.append(classification_report(ej, iej, output_dict=True)['1']['precision'])
        results.oiej.append(classification_report(ej, oiej, output_dict=True)['1']['precision'])
        results.oiej_dl.append(classification_report(ej, oiej_dl, output_dict=True)['1']['precision'])

    # plot results with markers, log (base=2) scale on x asis
    plt.figure(figsize=(8, 6))
    plt.plot(args.dims, results.iej, marker='o', label='IEJ')
    plt.plot(args.dims, results.oiej, marker='o', label='O_IEJ')
    plt.plot(args.dims, results.oiej_dl, marker='o', label='O_IEJ (DL)')
    plt.xlabel('Dimension')
    plt.ylabel('Precision')
    plt.xscale('log', base=2)
    plt.legend()

    plt.savefig(f'./results/{args.cmt}_{args.ckpt_type}.png')
