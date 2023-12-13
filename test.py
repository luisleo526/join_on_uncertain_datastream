from typing import Dict

import numpy as np
from prettytable import PrettyTable

from uncertain_object import UncertainObject
from utils import generator, timing, GENERATOR_TYPE


@timing
def timing_ej(d: int, n: int, samples: int = 20):
    result = []
    for _ in range(n):
        obj_1 = UncertainObject(generator(samples, [], [], [(x, 0, 1) for x in range(d)]))
        obj_2 = UncertainObject(generator(samples, [], [], [(x, 0, 1 + 0.5) for x in range(d)]))
        result.append(obj_1.ej(obj_2, 0.25))


@timing
def timing_iej(d: int, n: int, samples: int = 20):
    result = []
    for _ in range(n):
        obj_1 = UncertainObject(generator(samples, [], [], [(x, 0, 1) for x in range(d)]))
        obj_2 = UncertainObject(generator(samples, [], [], [(x, 0, 1 + 0.5) for x in range(d)]))
        result.append(obj_1.iej(obj_2, 0.25))


def accuracy_iej_weights(spec1: Dict[str, GENERATOR_TYPE],
                         spec2: Dict[str, GENERATOR_TYPE],
                         samples: int, eps: float, n: int = 1000):
    t = PrettyTable(['alpha', 'EJ', 'IEJ'])
    t.title = f'Accuracy @ eps = {eps:.4E}'
    for i in range(6):
        ej_passed = 0
        iej_passed = 0
        for _ in range(n):
            obj_1 = UncertainObject(generator(samples, **spec1))
            obj_2 = UncertainObject(generator(samples, **spec2))
            ej_passed += obj_1.ej(obj_2, eps)

            delta = (0.25 + 0.05 * i) * eps * np.ones(obj_1.num_dimensions)
            iej_passed += obj_1.iej(obj_2, eps, delta)

        t.add_row([0.25 + 0.05 * i, ej_passed, iej_passed])
    print(t)


def accuracy(spec1: Dict[str, GENERATOR_TYPE],
             spec2: Dict[str, GENERATOR_TYPE],
             samples: int, eps: float, n: int = 1000):
    ej_passed = 0
    iej_passed = 0
    o_iej_passed = 0
    for _ in range(n):
        obj_1 = UncertainObject(generator(samples, **spec1))
        obj_2 = UncertainObject(generator(samples, **spec2))
        ej_passed += obj_1.ej(obj_2, eps)
        iej_passed += obj_1.iej(obj_2, eps)
        o_iej_passed += obj_1.o_iej(obj_2, eps)

    t = PrettyTable(['EJ', 'IEJ', 'O_IEJ'])
    t.title = f'Accuracy @ eps = {eps:.4E}'
    t.add_row([ej_passed, iej_passed, o_iej_passed], divider=True)
    print(t)
    return t


dx = 100.0
dim = 2
alpha = 1.0
distribution_spec1 = dict(triangular=[], gaussian=[], uniform=[(x, 0.0, 1000.0) for x in range(dim)])
distribution_spec2 = dict(triangular=[], gaussian=[], uniform=[(x, -dx, 1000.0 + dx) for x in range(dim)])
accuracy_iej_weights(distribution_spec1, distribution_spec2, 100, 10)
