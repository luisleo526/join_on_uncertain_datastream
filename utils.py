from functools import wraps
from time import time
from typing import List, Tuple, Optional

import numpy as np
import numpy.random as npr

from uncertain_object import UncertainObject

GENERATOR_TYPE = List[Tuple[int, float, float]]


def generate_objects(num_objects, dim, threshold=None):
    objects = []
    means = np.random.uniform(-10, 10, (num_objects, dim))
    stds = np.random.uniform(1, 10, (num_objects, dim))
    epsilon = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=-1)
    if threshold is not None:
        epsilon = np.quantile(epsilon.flatten(), threshold)

    for i in range(num_objects):
        mean = means[i]
        std = stds[i]
        num_samples = np.random.randint(5, 60, 1)

        tri = []
        gau = []
        uni = []
        for d in range(dim):
            if d % 3 == 0:
                tri.append((d, mean[d] - std[d], mean[d] + std[d]))
            elif d % 3 == 1:
                gau.append((d, mean[d], std[d]))
            else:
                uni.append((d, mean[d] - std[d], mean[d] + std[d]))

        objects.append(UncertainObject(generator(num_samples, tri, gau, uni)))

    return objects, epsilon


def generator(num_samples: int,
              triangular: GENERATOR_TYPE, gaussian: GENERATOR_TYPE, uniform: GENERATOR_TYPE,
              constants: Optional[List[int]] = None
              ):
    """
    :param num_samples: Number of samples to generate
    :param triangular: List of tuples (id, min, max)
    :param gaussian: List of tuples (id, mean, std)
    :param uniform: List of tuples (id, min, max)
    :param constants: List of ids
    :return: List of samples
    """
    triangular_samples = [(x[0], npr.triangular(x[1], 0.5 * (x[1] + x[2]), x[2], size=num_samples)) for x in triangular]
    gaussian_samples = [(x[0], npr.normal(x[1], x[2], size=num_samples)) for x in gaussian]
    uniform_samples = [(x[0], npr.uniform(x[1], x[2], size=num_samples)) for x in uniform]

    all_samples = triangular_samples + gaussian_samples + uniform_samples
    all_samples.sort(key=lambda x: x[0])

    if constants is not None:
        all_samples += [(c, np.full(num_samples, c, dtype=float)) for c in constants]

    return np.stack([x[1] for x in all_samples]).T


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
              (f.__name__, args, kw, te - ts))
        return result

    return wrap
