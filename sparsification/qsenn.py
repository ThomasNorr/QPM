import numpy as np
import torch

from sparsification.glmBasedSparsification import compute_feature_selection_and_assignment


def compute_qsenn_feature_selection_and_assignment(model, train_loader, test_loader, log_folder, num_classes, seed,n_features, per_class = 5):
    feature_sel, sparse_matrices, biases, mean, std = compute_feature_selection_and_assignment(model, train_loader,
                                                                                               test_loader,
                                                                                               log_folder, num_classes, seed, n_features)
    weight_sparse, bias_sparse = get_sparsified_weights_for_factor(sparse_matrices[:-1], biases[:-1], per_class) # Last one in regularisation path has no regularisation
    print(f"Number of nonzeros in weight matrix: {torch.sum(weight_sparse != 0)}")
    return feature_sel, weight_sparse, bias_sparse, mean, std
def get_sparsified_weights_for_factor(weights, biases, factor,):
        no_reg_result_mat, no_reg_result_bias = weights[-1], biases[-1]
        goal_nonzeros = factor * no_reg_result_mat.shape[0]
        values = no_reg_result_mat.flatten()
        values = values[values != 0]
        values = -(torch.sort(-torch.abs(values))[0])
        if goal_nonzeros < len(values):
            threshold = (values[int(goal_nonzeros) - 1] + values[int(goal_nonzeros)]) / 2
        else:
            threshold = values[-1]
        max_val = torch.max(torch.abs(values))
        weight_sparse = discretize_2_bins_to_threshold(no_reg_result_mat, threshold, max_val)
        sel_idx = len(weights) - 1
        positive_weights_per_class = np.array(torch.sum(weight_sparse > 0, dim=1))
        negative_weights_per_class = np.array(torch.sum(weight_sparse < 0, dim=1))
        total_weight_count_per_class = positive_weights_per_class - negative_weights_per_class
        max_bias = torch.max(torch.abs(biases[sel_idx]))
        bias_sparse = torch.ones_like(biases[sel_idx]) * max_bias
        diff_n_weight = total_weight_count_per_class - np.min(total_weight_count_per_class)
        steps = np.max(diff_n_weight)
        single_step = 2 * max_bias / steps
        bias_sparse = bias_sparse - torch.tensor(diff_n_weight) * single_step
        bias_sparse = torch.clamp(bias_sparse, -max_bias, max_bias)
        return weight_sparse, bias_sparse


def  discretize_2_bins_to_threshold(data, treshold, max):
    boundaries = torch.tensor([-max, -treshold, treshold, max], device=data.device)
    bucketized_tensor = torch.bucketize(data, boundaries)
    means = torch.tensor([-max, 0, max], device=data.device)
    for i in range(len(means)):
        if means[i] == 0:
            break
        positive_index = int(len(means) / 2 + 1) + i
        positive_bucket = data[bucketized_tensor == positive_index + 1]
        negative_bucket = data[bucketized_tensor == i + 1]
        sum = 0
        total = 0
        for bucket in [positive_bucket, negative_bucket]:
            if len(bucket) == 0:
                continue
            sum += torch.sum(torch.abs(bucket))
            total += len(bucket)
        if total == 0:
            continue
        avg = sum / total
        means[i] = -avg
        means[positive_index] = avg
    discretized_tensor = means.cpu()[bucketized_tensor.cpu() - 1].to(bucketized_tensor.device)
    return discretized_tensor