from itertools import combinations
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from uncertain_object import UncertainObject
from utils import generator

if __name__ == '__main__':

    objects: List[UncertainObject] = []

    dims = int(input("Enter the number of dimensions: "))
    num_objects = int(input("Enter the number of objects: "))
    epsilon = float(input("Enter the value of epsilon: "))
    total_tests = int(num_objects * (num_objects - 1) / 2)
    for _ in range(num_objects):
        mean = np.random.randint(100, 150, 1)
        std = np.random.randint(1, 11, 1)
        num_samples = np.random.randint(30, 100, 1)
        objects.append(UncertainObject(
            generator(num_samples, [], [],
                      [(d, mean - (d + 1) * std, mean + (d + 1) * std) for d in range(dims)]))
        )

    prune_power = np.zeros((3, 10))
    betas = np.linspace(0, 0.5, 10)
    for i, j in tqdm(combinations(objects, 2), total=total_tests):
        for k in range(10):
            prune_power[0, k] += i.ej(j, epsilon, beta=betas[k])
            prune_power[1, k] += i.iej(j, epsilon, beta=betas[k])
            prune_power[2, k] += i.o_iej(j, epsilon, beta=betas[k])

    prune_power /= total_tests * 100
    prune_power = 1.0 - prune_power
    prune_power *= 100

    plt.figure()
    plt.plot(betas, prune_power[0, :], label='EJ', marker='o')
    plt.plot(betas, prune_power[1, :], label='IEJ', marker='o')
    plt.plot(betas, prune_power[2, :], label='O_IEJ', marker='o')
    plt.legend()
    plt.xlabel('Beta')
    plt.ylabel('Prune power (%)')

    # save figure
    plt.savefig(f'prune_power_{dims}-{num_objects}-{epsilon}.png', dpi=180, bbox_inches='tight')
