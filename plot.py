import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset import UncertainObjectDataset
from model import IEJModel
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

if __name__ == '__main__':

    num_objects = int(input('Number of objects: '))
    num_turns = int(input('Number of turns: '))

    total_tests = int(num_objects * (num_objects - 1) / 2)
    dims = np.array(list(range(1, 9)), dtype=int)
    dims = np.power(2, dims)

    results = np.zeros((4, dims.size, total_tests * num_turns), dtype=int)

    data = {
        'precision': np.zeros((6, dims.size)),
        'recall': np.zeros((6, dims.size)),
        'f1-score': np.zeros((6, dims.size))
    }

    for dim_idx in range(dims.size):

        dim = dims[dim_idx]

        objects, _ = generate_objects(num_objects, dim, None)
        eval_ds = UncertainObjectDataset(num_objects, dim, [0.1 + 0.025 * i for i in range(30)])

        model = IEJModel(dim, 4)
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

        logger.info(f'Dim {dim}')
        logger.info('IEJ')
        logger.info('\n' + classification_report(ej, iej, digits=4))
        logger.info('O_IEJ')
        logger.info('\n' + classification_report(ej, oiej, digits=4))
        logger.info('O_IEJ (DL)')
        logger.info('\n' + classification_report(ej, oiej_dl, digits=4))

        iej = classification_report(ej, iej, output_dict=True)
        oiej = classification_report(ej, oiej, output_dict=True)
        oiej_dl = classification_report(ej, oiej_dl, output_dict=True)

        for tgt_idx, tgt in enumerate(['0', '1']):
            for metric in ['precision', 'recall', 'f1-score']:
                data[metric][0 + 3 * tgt_idx, dim_idx] = iej[tgt][metric]
                data[metric][1 + 3 * tgt_idx, dim_idx] = oiej[tgt][metric]
                data[metric][2 + 3 * tgt_idx, dim_idx] = oiej_dl[tgt][metric]

    fig, axs = plt.subplots(3, 2, figsize=(10, 13))

    axs[0][0].plot(dims, data['precision'][0], label='IEJ', marker='o')
    axs[0][0].plot(dims, data['precision'][1], label='O_IEJ', marker='o')
    axs[0][0].plot(dims, data['precision'][2], label='O_IEJ (DL)', marker='o')
    axs[0][0].set_title('Precision (0)')
    axs[0][0].set_xscale('log', base=2)
    axs[0][0].legend()

    axs[1][0].plot(dims, data['recall'][0], label='IEJ', marker='o')
    axs[1][0].plot(dims, data['recall'][1], label='O_IEJ', marker='o')
    axs[1][0].plot(dims, data['recall'][2], label='O_IEJ (DL)', marker='o')
    axs[1][0].set_title('Recall (0)')
    axs[1][0].set_xscale('log', base=2)
    axs[1][0].legend()

    axs[2][0].plot(dims, data['f1-score'][0], label='IEJ', marker='o')
    axs[2][0].plot(dims, data['f1-score'][1], label='O_IEJ', marker='o')
    axs[2][0].plot(dims, data['f1-score'][2], label='O_IEJ (DL)', marker='o')
    axs[2][0].set_title('F1 Score (0)')
    axs[2][0].set_xscale('log', base=2)
    axs[2][0].legend()

    axs[0][1].plot(dims, data['precision'][3], label='IEJ', marker='o')
    axs[0][1].plot(dims, data['precision'][4], label='O_IEJ', marker='o')
    axs[0][1].plot(dims, data['precision'][5], label='O_IEJ (DL)', marker='o')
    axs[0][1].set_title('Precision (1)')
    axs[0][1].set_xscale('log', base=2)
    axs[0][1].legend()

    axs[1][1].plot(dims, data['recall'][3], label='IEJ', marker='o')
    axs[1][1].plot(dims, data['recall'][4], label='O_IEJ', marker='o')
    axs[1][1].plot(dims, data['recall'][5], label='O_IEJ (DL)', marker='o')
    axs[1][1].set_title('Recall (1)')
    axs[1][1].set_xscale('log', base=2)
    axs[1][1].legend()

    axs[2][1].plot(dims, data['f1-score'][3], label='IEJ', marker='o')
    axs[2][1].plot(dims, data['f1-score'][4], label='O_IEJ', marker='o')
    axs[2][1].plot(dims, data['f1-score'][5], label='O_IEJ (DL)', marker='o')
    axs[2][1].set_title('F1 Score (1)')
    axs[2][1].set_xscale('log', base=2)
    axs[2][1].legend()

    # save figure
    plt.savefig(f'{num_objects}.png', dpi=180, bbox_inches='tight')
