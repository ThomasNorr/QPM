from pathlib import Path

import numpy as np
import torch

from dataset_classes.cub200 import CUB200Class


def get_structural_grounding_for_weight_matrix(rel_weight):
    cross_class_sim = get_cross_class_similarity(rel_weight)
    base_folder = Path.home() / "tmp" / "Datasets" / "CUB200"
    class_sim_gt = CUB200Class.get_class_sim(base_folder)
    cross_class_sim = cross_class_sim.cpu().numpy()
    answer = get_top_x_similar(cross_class_sim, class_sim_gt, [50])[0]
    return answer

def get_top_x_similar(cross_sim, gt_sim, xs):
    cross_sim_flat = cross_sim.flatten()
    gt_sim_flat = gt_sim.flatten()
    sorted_indices = np.argsort(gt_sim_flat)
    sorted_indices = sorted_indices[::-1]
    answer = []
    for x in xs:
        top_x_indices = sorted_indices[:x]
        total_cross = np.sum(cross_sim_flat[top_x_indices])
        total_gt = np.sum(gt_sim_flat[top_x_indices])
        frac = total_cross / total_gt
        answer.append(frac)
    return answer

def get_cross_class_similarity(weight):
    # weight = model.model.linear.weight
    weight = weight / torch.norm(weight, dim=1, keepdim=True)
    sim = torch.matmul(weight, weight.T)
    sim[torch.eye(sim.shape[0]).bool()] = 0
    return sim