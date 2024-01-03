import random
from itertools import combinations
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import generate_objects


class UncertainObjectDataset(Dataset):
    def __init__(self, num_objects: int, dim: int, threshold_choices: List[float]):
        self.objects, self.distance_matrix = generate_objects(num_objects, dim, None)
        self.epsilons = []
        for threshold in threshold_choices:
            self.epsilons.append(np.quantile(self.distance_matrix.flatten(), threshold))
        self.num_objects = num_objects
        self.indices = list(combinations(range(num_objects), 2))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        return self.objects[i], self.objects[j], random.choice(self.epsilons)


def collate_fn(batch):
    a_tensor = []
    b_tensor = []
    epsilons = []
    min_distances = []
    for a, b, epsilon in batch:
        a_tensor.append(a.mbr_tensor)
        b_tensor.append(b.mbr_tensor)
        epsilons.append(epsilon)
        min_distances.append(a % b)

    return torch.stack(a_tensor), torch.stack(b_tensor), torch.tensor(epsilons), torch.tensor(min_distances)
