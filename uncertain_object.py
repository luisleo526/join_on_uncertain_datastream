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

        # prob = np.zeros(self.num_dimensions)
        #
        # area_self = self.mbr.max - self.mbr.min + 2.0 * delta
        # area_other = other.mbr.max - other.mbr.min + 2.0 * delta
        #
        # for i in range(self.num_dimensions):
        #
        #     if self.mbr.max[i] > other.mbr.max[i]:
        #         if self.mbr.min[i] < other.mbr.min[i]:
        #             area = other.mbr.max[i] - other.mbr.min[i] + 2.0 * delta[i]
        #             prob[i] = area / area_self[i]
        #         else:
        #             area = other.mbr.max[i] - self.mbr.min[i] + 2.0 * delta[i]
        #             if area < 0.0:
        #                 area = 0.0
        #             prob[i] = area ** 2 / (area_self[i] * area_other[i])
        #
        #     else:
        #         if other.mbr.min[i] < self.mbr.min[i]:
        #             area = self.mbr.max[i] - self.mbr.min[i] + 2.0 * delta[i]
        #             prob[i] = area / area_other[i]
        #         else:
        #             area = self.mbr.max[i] - other.mbr.min[i] + 2.0 * delta[i]
        #             if area < 0.0:
        #                 area = 0.0
        #             prob[i] = area ** 2 / (area_self[i] * area_other[i])

        area_self = self.mbr.max - self.mbr.min + 2.0 * delta
        area_other = other.mbr.max - other.mbr.min + 2.0 * delta

        # Calculate the conditions
        cond_self_greater = self.mbr.max > other.mbr.max
        cond_self_smaller = self.mbr.min < other.mbr.min

        # Calculate the areas based on conditions
        area1 = np.where(cond_self_smaller,
                         other.mbr.max - other.mbr.min + 2.0 * delta,
                         other.mbr.max - self.mbr.min + 2.0 * delta)
        area1 = np.maximum(area1, 0.0)  # to ensure area is not negative

        area2 = np.where(cond_self_smaller,
                         self.mbr.max - other.mbr.min + 2.0 * delta,
                         self.mbr.max - self.mbr.min + 2.0 * delta)
        area2 = np.maximum(area2, 0.0)  # to ensure area is not negative

        # Calculate probabilities based on conditions
        prob = np.where(cond_self_greater,
                        np.where(cond_self_smaller, area1 / area_self, area1 ** 2 / (area_self * area_other)),
                        np.where(cond_self_smaller, area2 ** 2 / (area_self * area_other), area2 / area_other))

        assert np.all(prob <= 1.0)

        return np.prod(prob) > beta

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

        # std = self.mbr.std + other.mbr.std
        std = np.sqrt(np.square(self.mbr.std) + np.square(other.mbr.std))
        # std = np.maximum(self.mbr.std, other.mbr.std)
        delta = 1.0 / (2.0 * np.linalg.norm(std, ord=1)) * eps * std
        return self._check_overlapping(other, delta, beta)
