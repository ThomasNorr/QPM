import torch


def softmax_feature_maps(x):
    # done: verify that this applies softmax along first dimension
    return torch.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)