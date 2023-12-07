from uncertain_object import UncertainObject
from utils import generator, timing


@timing
def test_epsilon_join(d: int, n: int, samples: int = 20):
    result = []
    for _ in range(n):
        obj_1 = UncertainObject(generator(samples, [], [], [(x, 0, 1) for x in range(d)]))
        obj_2 = UncertainObject(generator(samples, [], [], [(x, 0, 1 + 0.5) for x in range(d)]))
        result.append(obj_1.ej(obj_2, 0.25))


@timing
def test_increased_epsilon_join(d: int, n: int, samples: int = 20):
    result = []
    for _ in range(n):
        obj_1 = UncertainObject(generator(samples, [], [], [(x, 0, 1) for x in range(d)]))
        obj_2 = UncertainObject(generator(samples, [], [], [(x, 0, 1 + 0.5) for x in range(d)]))
        result.append(obj_1.iej(obj_2, 0.25))