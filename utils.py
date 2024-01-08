from functools import wraps
from time import time
from typing import List, Tuple, Optional

import numpy as np
import numpy.random as npr

from uncertain_object import UncertainObject

GENERATOR_TYPE = List[Tuple[int, float, float]]


def time_series(t, d):
    phase = 2 * np.pi / 18 * d
    mean = np.sin(t + phase)
    mean += 0.25 * np.sin(2 * t + phase) * np.cos(3 * t - phase)
    mean += 0.125 * np.sin(5 * t + phase) * np.cos(7 * t - phase)
    mean += 0.0625 * np.sin(11 * t + phase) * np.cos(13 * t - phase)
    return mean * 10


def generate_time_streams(num_objects, dim):
    objects = []
    t = np.random.uniform(0, 4 * np.pi, num_objects)
    t = np.sort(t)
    means = np.stack([time_series(t, d) for d in range(dim)]).T
    stds = np.random.uniform(1, 3, (num_objects, dim))

    return generate_objects(num_objects, dim, means, stds)


def generate_objects(num_objects, dim, means=None, stds=None):
    objects = []
    if means is None:
        means = np.random.uniform(-10, 10, (num_objects, dim))
    if stds is None:
        stds = np.random.uniform(1, 5, (num_objects, dim))

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

    return objects


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
