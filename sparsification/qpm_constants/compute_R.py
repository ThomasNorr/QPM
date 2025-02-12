import torch

from evaluation.Metrics.Correlation import sim_matrix


def compute_cos_sim_matrix(feat_class_matrix):
    feat_class_matrix = torch.tensor(feat_class_matrix)
    return sim_matrix(feat_class_matrix, feat_class_matrix)