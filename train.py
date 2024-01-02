from itertools import combinations

import torch
from tqdm import tqdm

from model import IEJModel
from utils import generate_objects

if __name__ == '__main__':
    num_objects = 30
    dim = 5
    total_tests = int(num_objects * (num_objects - 1) / 2)
    model = IEJModel(dim, 4)
    objects, epsilon = generate_objects(num_objects, dim, 0.2)

    for idx, (a, b) in enumerate(tqdm(combinations(objects, 2), total=total_tests)):
        w = model(a.mbr_tensor, b.mbr_tensor)
        a_min, a_max, a_std, a_mean = torch.unbind(a.mbr_tensor, dim=1)
        b_min, b_max, b_std, b_mean = torch.unbind(b.mbr_tensor, dim=1)

        
