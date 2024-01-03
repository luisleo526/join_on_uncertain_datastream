import torch


def iej_loss(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon: torch.Tensor, min_distance: torch.Tensor):
    batch_size, num_dimension = w.shape

    a_min, a_max, a_std, a_mean = torch.unbind(a, dim=2)
    b_min, b_max, b_std, b_mean = torch.unbind(b, dim=2)
    delta: torch.Tensor = w * epsilon.unsqueeze(1).expand(batch_size, num_dimension) * 0.5

    a_max = a_max + delta
    a_min = a_min - delta
    b_max = b_max + delta
    b_min = b_min - delta

    area_a = a_max - a_min
    area_b = b_max - b_min

    cond_a_greater: torch.Tensor = a_max > b_max
    cond_a_smaller: torch.Tensor = a_min < b_min

    # a_max > b_max
    area1 = torch.where(cond_a_smaller, b_max - b_min, b_max - a_min)
    area1 = torch.clamp(area1, min=0.0)

    # a_max < b_max
    area2 = torch.where(cond_a_smaller, a_max - b_min, a_max - a_min)
    area2 = torch.clamp(area2, min=0.0)

    prob = torch.where(cond_a_greater,
                       torch.where(cond_a_smaller, area1 / area_a, area1 ** 2 / (area_a * area_b)),
                       torch.where(cond_a_smaller, area2 ** 2 / (area_a * area_b), area2 / area_b))
    prob = torch.prod(prob, dim=1)

    overlapped = torch.where(prob > 0.0, 1, -1)

    # a_max > b_max
    area1 = torch.where(cond_a_smaller, b_max - b_min, b_max - a_min)

    # a_max < b_max
    area2 = torch.where(cond_a_smaller, a_max - b_min, a_max - a_min)

    distance = torch.where(cond_a_greater, area1, area2)
    distance = torch.sqrt(torch.sum(torch.square(distance), dim=1))

    sign = torch.where(epsilon > min_distance, 1, -1)

    lambda_1 = 1.0
    lambda_2 = 1.0

    loss = 0.5 * overlapped * (-lambda_1 * (1 + sign) * prob + lambda_2 * (1 - sign) * distance)

    return loss.mean()
