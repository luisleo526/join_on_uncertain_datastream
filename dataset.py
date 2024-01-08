from itertools import combinations

import torch
from torch.utils.data import Dataset

from utils import generate_objects


class UncertainObjectDataset(Dataset):
    def __init__(self, num_objects: int, dim: int):
        self.objects = generate_objects(num_objects, dim)
        self.epsilons = []
        self.num_objects = num_objects
        self.indices = list(combinations(range(num_objects), 2))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        return self.objects[i], self.objects[j]


def collate_fn(batch):
    a_tensor = []
    b_tensor = []
    epsilons = []
    min_distances = []
    for a, b in batch:
        a_tensor.append(a.mbr_tensor)
        b_tensor.append(b.mbr_tensor)
        epsilons.append((a % b) * 0.95)
        min_distances.append(a % b)

        a_tensor.append(a.mbr_tensor)
        b_tensor.append(b.mbr_tensor)
        epsilons.append((a % b) * 1.05)
        min_distances.append(a % b)

    return torch.stack(a_tensor), torch.stack(b_tensor), torch.tensor(epsilons), torch.tensor(min_distances)


def collate_fn2(batch):
    a_tensor = []
    b_tensor = []

    epsilons = []
    a_objects = []
    b_objects = []
    for a, b in batch:
        a_tensor.append(a.mbr_tensor)
        b_tensor.append(b.mbr_tensor)
        epsilons.append((a % b) * 0.95)
        a_objects.append(a)
        b_objects.append(b)

        a_tensor.append(a.mbr_tensor)
        b_tensor.append(b.mbr_tensor)
        epsilons.append((a % b) * 1.05)
        a_objects.append(a)
        b_objects.append(b)

    a_tensor = torch.stack(a_tensor)
    b_tensor = torch.stack(b_tensor)

    return a_tensor, b_tensor, epsilons, a_objects, b_objects
