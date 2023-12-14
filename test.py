from itertools import combinations
from typing import List

import numpy as np
from tqdm import tqdm

from uncertain_object import UncertainObject
from utils import generator

if __name__ == '__main__':

    objects: List[UncertainObject] = []
    dims = 5
    num_objects = 300
    for _ in range(num_objects):
        mean = np.random.randint(100, 130, 1)
        std = np.random.randint(1, 11, 1)
        num_samples = np.random.randint(30, 100, 1)
        objects.append(UncertainObject(
            generator(num_samples, [], [], [(d, mean - std, mean + std) for d in range(dims)]))
        )

    ej_passed = 0
    iej_passed = 0
    o_iej_passed = 0
    for i, j in tqdm(combinations(objects, 2), total=int(num_objects * (num_objects - 1) / 2)):
        ej_passed += i.ej(j, 1.0, beta=0.0)
        iej_passed += i.iej(j, 1.0, beta=0.0)
        o_iej_passed += i.o_iej(j, 1.0, beta=0.0)

    print(f'ej_passed: {ej_passed}')
    print(f'iej_passed: {iej_passed}')
    print(f'o_iej_passed: {o_iej_passed}')
