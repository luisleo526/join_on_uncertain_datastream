import numpy as np
from munch import Munch


class UncertainObject(object):

    def __init__(self, samples: np.ndarray):
        """
        :param samples: (num_samples, num_dimensions) array of samples
        """
        self.samples = samples
        self.num_samples = samples.shape[0]
        self.num_dimensions = samples.shape[1]

    @property
    def mbr(self):
        """
        :return: Munch(min, max, std) where min, max, std are (num_dimensions,) arrays
        """
        return Munch(min=np.min(self.samples, axis=0),
                     max=np.max(self.samples, axis=0),
                     std=np.std(self.samples, axis=0))

    def ej(self, other, eps: float) -> bool:
        """
        :param other: UncertainObject
        :param eps: Epsilon
        :return: Boolean
        """
        assert self.num_dimensions == other.num_dimensions

        # Compute the distance between each pair of samples
        distances = np.linalg.norm(self.samples[:, None, :] - other.samples[None, :, :], axis=-1)
        return np.any(distances < eps)

    def _check_overlapping(self, other, delta: np.ndarray) -> bool:
        """
        :param other: UncertainObject
        :param delta: (num_dimensions,) array of deltas
        :return: Boolean
        """
        for i in np.random.permutation(self.num_dimensions):
            # check if two mbr overlap in dimension I after expanding by delta
            if (self.mbr.min[i] - delta[i] > other.mbr.max[i] + delta[i] or
                    self.mbr.max[i] + delta[i] < other.mbr.min[i] - delta[i]):
                return False
        return True

    def iej(self, other, eps: float) -> bool:
        """
        :param other: UncertainObject
        :param eps: Epsilon
        :return: Boolean
        """
        assert self.num_dimensions == other.num_dimensions

        delta = 1.0 / (2 * np.sqrt(self.num_dimensions)) * eps * np.ones(self.num_dimensions)

        return self._check_overlapping(other, delta)

    def o_iej(self, other, eps: float) -> bool:
        """
        :param other: UncertainObject
        :param eps: Epsilon
        :return: Boolean
        """
        assert self.num_dimensions == other.num_dimensions

        std = self.mbr.std + other.mbr.std
        delta = 1.0 / (2.0 * np.sqrt(np.sum(np.square(std)))) * eps * std

        return self._check_overlapping(other, delta)
