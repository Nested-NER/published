from random import sample

import numpy as np
import torch
from torch import optim


def create_opt(parameters, opt, lr):
    if opt == "Adam":
        return optim.Adam(parameters, lr=lr)
    return None


def adjust_learning_rate(optimizer, decay_rate=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate
    print(f"lr change to {optimizer.param_groups[0]['lr']}")


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # vec is only 1d vec
    # return the argmax as a python int
    _, idx = torch.max(vec, 0)
    return to_scalar(idx)


def log_sum_exp(vec_list):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm
    vec is n * m, norm in row
    return n * 1
    """
    if type(vec_list) == list:
        mat = torch.stack(vec_list, 1)
    else:
        mat = vec_list
    row, column = mat.size()
    ret_l = []
    for i in range(row):
        vec = mat[i]
        max_score = vec[argmax(vec)]
        max_score_broadcast = max_score.view(-1).expand(1, vec.size()[0])
        ret_l.append(max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))
    return torch.cat([item.expand(1) for item in ret_l], 0)


def check_contain(p1, p2):
    """
    if p1 contain p2
        return True
    """
    # if type(p1) != "tuple":
    #     p1 = tuple(p1)
    # if type(p2) != "tuple":
    #     p2 = tuple(p2)
    # left = min(p1[0], p2[0])
    # right = max(p1[1], p2[1])
    # union = (left, right)
    # return p1 == union
    return p1[0] <= p2[0] and p1[1] >= p2[1]


def check_overlap(p1, p2):
    return (p1[0] - p2[1]) * (p2[0] - p1[1]) > 0 and (p1[0] - p2[0]) * (p1[1] - p2[1]) > 0


def calc_iou(b1, b2):
    assert b1.size > 0
    if b2.size == 0:
        return np.zeros((b1.shape[0], 1))
    l_max = np.maximum(b1[:, 0].reshape((-1, 1)), b2[:, 0].reshape((1, -1)))
    l_min = np.minimum(b1[:, 0].reshape((-1, 1)), b2[:, 0].reshape((1, -1)))
    r_max = np.maximum(b1[:, 1].reshape((-1, 1)), b2[:, 1].reshape((1, -1)))
    r_min = np.minimum(b1[:, 1].reshape((-1, 1)), b2[:, 1].reshape((1, -1)))

    inter = np.maximum(0, r_min - l_max)
    union = r_max - l_min
    return inter / union


def select_toi(toi_batch):
    tois = [np.array([(t[0], t[1]) for t in sent]) for sent in toi_batch]
    labels = [np.array([t[2] for t in sent]) for sent in toi_batch]

    return tois, labels


def find_boundary(start, boundary):
    for i, b in enumerate(boundary):
        if b > start:
            return i
    return len(boundary)


def sequent_mask(sent_len, max_sent_len):
    mask = []
    for num in sent_len:
        m = [1] * num
        m.extend([0] * (max_sent_len - num))
        mask.append(m)
    return torch.from_numpy(np.array(mask)) == 1


def batch_split(score, toi_section):
    score = score.cpu()
    result = [torch.index_select(score, 0, torch.arange(0, toi_section[0]).long())]
    for i in range(len(toi_section) - 1):
        result.append(torch.index_select(score, 0, torch.arange(toi_section[i], toi_section[i + 1]).long()))
    return result


def generate_mask(shape):
    return torch.from_numpy(np.ones((shape[0], shape[1], shape[1])))


if __name__ == "__main__":
    # print(check_contain((3, 7), (5, 7)))
    # print(check_contain((0, 5), (0, 6)))
    print(check_overlap((1, 5), (5, 7)))
    print(check_overlap((1, 5), (4, 7)))
    print(check_overlap((1, 5), (0, 7)))
    print(check_overlap((1, 5), (1, 7)))
    print(check_overlap((1, 5), (0, 4)))
    print("")
    print(check_overlap((5, 7), (1, 5)))
    print(check_overlap((4, 7), (1, 5)))
    print(check_overlap((0, 7), (1, 5)))
    print(check_overlap((1, 7), (1, 5)))
    print(check_overlap((0, 4), (1, 5)))
