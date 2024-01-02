from itertools import combinations

from tqdm import tqdm

from model import IEJModel
from utils import generate_objects

if __name__ == '__main__':
    num_objects = 30
    dim = 3
    total_tests = int(num_objects * (num_objects - 1) / 2)
    model = IEJModel(dim, 4)
    objects, epsilon = generate_objects(num_objects, dim, 0.2)

    for idx, (i, j) in enumerate(tqdm(combinations(objects, 2), total=total_tests)):
          pass
    # print(objects[0].mbr_tensor.shape)
