import torch


def compute_real_gt_max(features,  weights, labels):
    no_min_features = features - features.min(dim=0)[0]
    no_min_denominator = no_min_features.sum(dim=0)
    no_min_features_part = torch.zeros((weights.shape[0], features.shape[1],))
    for unique_class in labels.unique():
        gt_mask = labels == unique_class
        upper_nominator = no_min_features[gt_mask].sum(dim=0)
        no_min_features_part[unique_class] = upper_nominator / torch.clamp(no_min_denominator, min=1e-8)
    no_min_max_per_feature = no_min_features_part.max(dim=0)[0]
    return  1 - no_min_max_per_feature.mean().item()