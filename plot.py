import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset import UncertainObjectDataset
from model import IEJModel
from utils import generate_objects

if __name__ == '__main__':

    num_objects = int(input('Number of objects: '))
    num_turns = int(input('Number of turns: '))

    total_tests = int(num_objects * (num_objects - 1) / 2)
    dims = np.array(list(range(1, 8)), dtype=int)
    dims = np.power(2, dims)

    results = np.zeros((4, dims.size, total_tests * num_turns), dtype=int)

    for dim_idx in range(dims.size):

        dim = dims[dim_idx]

        objects, _ = generate_objects(num_objects, dim, None)
        eval_ds = UncertainObjectDataset(num_objects, dim, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        model = IEJModel(dim, 4)
        model.load_state_dict(torch.load(f'./ckpt/iej_{dim}_best.pth'))
        model.eval()

        with torch.no_grad():
            for turn in range(num_turns):
                for idx in tqdm(range(len(eval_ds)), desc=f'Dim {dim} Turn {turn}'):
                    a, b, epsilon = eval_ds[idx]
                    w = model(a.mbr_tensor.unsqueeze(0), b.mbr_tensor.unsqueeze(0)).numpy()[0]
                    delta = w * 0.5 * epsilon

                    results[0, dim_idx, turn * total_tests + idx] = a.ej(b, epsilon)
                    results[1, dim_idx, turn * total_tests + idx] = a.iej(b, epsilon)
                    results[2, dim_idx, turn * total_tests + idx] = a.o_iej(b, epsilon)
                    results[3, dim_idx, turn * total_tests + idx] = a.iej(b, epsilon, delta)

    data = {
        'precision': np.zeros((6, dims.size)),
        'recall': np.zeros((6, dims.size)),
        'f1-score': np.zeros((6, dims.size))
    }

    for k in range(dims.size):
        iej = classification_report(results[0, k], results[1, k], output_dict=True, zero_division=0.0)
        oiej = classification_report(results[0, k], results[2, k], output_dict=True, zero_division=0.0)
        dl_iej = classification_report(results[0, k], results[3, k], output_dict=True, zero_division=0.0)

        for tgt_idx, tgt in enumerate(['0', '1']):
            for metric in ['precision', 'recall', 'f1-score']:
                data[metric][0 + 3 * tgt_idx, k] = iej[tgt][metric]
                data[metric][1 + 3 * tgt_idx, k] = oiej[tgt][metric]
                data[metric][2 + 3 * tgt_idx, k] = dl_iej[tgt][metric]

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
