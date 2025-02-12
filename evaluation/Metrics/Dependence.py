import torch


def compute_class_independence(features, weights, labels,):
    feature_class_part = torch.zeros((weights.shape[0], features.shape[1],))
    denominator = torch.relu(features).sum(dim=0)
    no_min_features = features - features.min(dim=0)[0]
    no_min_denominator = no_min_features.sum(dim=0)
    no_min_features_part = torch.zeros_like(feature_class_part)
    for unique_class in labels.unique():
        gt_mask = labels == unique_class
        upper_nominator = torch.relu(features[gt_mask]).sum(dim=0)
        feature_class_part[unique_class] = upper_nominator / torch.clamp(denominator, min=1e-8)

        upper_nominator = no_min_features[gt_mask].sum(dim=0)
        no_min_features_part[unique_class] = upper_nominator / torch.clamp(no_min_denominator, min=1e-8)
    no_min_max_per_feature = no_min_features_part.max(dim=0)[0]
    print("Max CI bef 1- ", no_min_max_per_feature.max().item())
    return 1 - no_min_max_per_feature.mean().item()




def compute_contribution_top_feature(features, outputs, weights,  labels):
    with torch.no_grad():
        total_pre_softmax, predicted_classes = torch.max(outputs, dim=1)
        feature_part = features * weights.to(features.device)[predicted_classes]
        class_specific_feature_part = torch.zeros((weights.shape[0], features.shape[1],))
        feature_class_part = torch.zeros((weights.shape[0], features.shape[1],))
        for unique_class in predicted_classes.unique():
            mask = predicted_classes == unique_class
            class_specific_feature_part[unique_class] = feature_part[mask].mean(dim=0)
            gt_mask = labels == unique_class
            feature_class_part[unique_class] = feature_part[gt_mask].mean(dim=0)
        abs_features = feature_part.abs()
        abs_sum = abs_features.sum(dim=1)
        fractions_abs = abs_features / abs_sum[:, None]
        abs_max = fractions_abs.max(dim=1)[0]
        mask = ~torch.isnan(abs_max)
        abs_max = abs_max[mask]
    return abs_max.mean()