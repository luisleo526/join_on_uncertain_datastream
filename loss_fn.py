import torch


def iej_loss(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon: float):
    a_min, a_max, a_std, a_mean = torch.unbind(a, dim=1)
    b_min, b_max, b_std, b_mean = torch.unbind(b, dim=1)
    delta: torch.Tensor = w * epsilon * 0.5

    area_a = a_max - a_min + 2.0 * delta
    area_b = b_max - b_min + 2.0 * delta

    cond_a_greater: torch.Tensor = a_max > b_max
    cond_a_smaller: torch.Tensor = a_min < b_min

    area1 = torch.where(cond_a_smaller,
                        b_max - b_min + 2.0 * delta,
                        b_max - a_min + 2.0 * delta)

    _area1 = torch.clamp(area1, min=0.0)

    area2 = torch.where(cond_a_smaller,
                        a_max - b_min + 2.0 * delta,
                        a_max - a_min + 2.0 * delta)

    _area2 = torch.clamp(area2, min=0.0)

    prob = torch.where(cond_a_greater,
                       torch.where(cond_a_smaller, area1 / area_a, area1 ** 2 / (area_a * area_b)),
                       torch.where(cond_a_smaller, area2 ** 2 / (area_a * area_b), area2 / area_b))

    _prob = torch.where(cond_a_greater,
                        torch.where(cond_a_smaller, _area1 / area_a, _area1 ** 2 / (area_a * area_b)),
                        torch.where(cond_a_smaller, _area2 ** 2 / (area_a * area_b), _area2 / area_b))

    assert torch.all(_prob <= 1.0), f'Probabilities must be less than or equal to 1.0, but got {_prob}'

    overlapped = torch.prod(_prob) > 0.0





