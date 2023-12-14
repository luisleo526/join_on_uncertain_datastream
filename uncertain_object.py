from typing import Optional

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

    def ej(self, other, eps: float, beta: float = 0.0) -> bool:
        """
        :param other: UncertainObject
        :param eps: Epsilon
        :param beta: Probability threshold
        :return: Boolean
        """
        assert self.num_dimensions == other.num_dimensions

        # Compute the distance between each pair of samples
        distances = np.linalg.norm(self.samples[:, None, :] - other.samples[None, :, :], axis=-1)

        return distances[distances < eps].size > beta * distances.size

    def _check_overlapping(self, other, delta: np.ndarray, beta: float = 0.0) -> bool:
        """
        :param other: UncertainObject
        :param delta: (num_dimensions,) array of deltas
        :param beta: Probability threshold
        :return: Boolean
        """

        assert np.all(delta > 0.0)

        area = np.where(self.mbr.max > other.mbr.max,
                        np.where(self.mbr.min > other.mbr.min,
                                 other.mbr.max - self.mbr.min + 2.0 * delta,
                                 other.mbr.max - other.mbr.min + 2.0 * delta
                                 ),
                        np.where(self.mbr.min > other.mbr.min,
                                 self.mbr.max - self.mbr.min + 2.0 * delta,
                                 self.mbr.max - other.mbr.min + 2.0 * delta
                                 )
                        )

        area = np.where(area > 0.0, area, 0.0)

        prob = (np.prod(area) ** 2 / np.prod(self.mbr.max - self.mbr.min + 2.0 * delta) /
                np.prod(other.mbr.max - other.mbr.min + 2.0 * delta))

        assert prob <= 1.0

        return prob > beta

    def iej(self, other, eps: float, delta: Optional[np.ndarray] = None, beta: float = 0.0) -> bool:
        """
        :param other: UncertainObject
        :param eps: Epsilon
        :param delta: (num_dimensions,) array of deltas
        :param beta: Probability threshold
        :return: Boolean
        """
        assert self.num_dimensions == other.num_dimensions

        if delta is None:
            delta = 1.0 / (2.0 * np.sqrt(self.num_dimensions)) * eps * np.ones(self.num_dimensions)

        return self._check_overlapping(other, delta, beta)

    def o_iej(self, other, eps: float, beta: float = 0.0) -> bool:
        """
        :param other: UncertainObject
        :param eps: Epsilon
        :param beta: Probability threshold
        :return: Boolean
        """
        assert self.num_dimensions == other.num_dimensions

        std = self.mbr.std + other.mbr.std
        delta = 1.0 / (2.0 * np.linalg.norm(std, ord=1)) * eps * std
        return self._check_overlapping(other, delta, beta)
