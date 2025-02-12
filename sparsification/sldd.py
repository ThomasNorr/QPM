import numpy as np
import torch

from sparsification.glmBasedSparsification import compute_feature_selection_and_assignment


def compute_sldd_feature_selection_and_assignment(model, train_loader, test_loader, log_folder, num_classes, seed,
                                                   per_class=5, select_features=50):
    feature_sel, sparse_matrices, biases, mean, std = compute_feature_selection_and_assignment(model, train_loader,
                                                                                               test_loader,
                                                                                               log_folder, num_classes,
                                                                                               seed, select_features=select_features)
    weight_sparse, bias_sparse = get_sparsified_weights_for_factor(sparse_matrices,biases,
                                                                   per_class)  # Last one in regularisation path has none
    return feature_sel, weight_sparse, bias_sparse, mean, std

def get_sparsified_weights_for_factor(sparse_layer,biases,keep_per_class, drop_rate=0.5):
    nonzero_entries = [torch.sum(torch.count_nonzero(sparse_layer[i])) for i in range(len(sparse_layer))]
    mean_sparsity = np.array([nonzero_entries[i] / sparse_layer[i].shape[0] for i in range(len(sparse_layer))])
    factor =keep_per_class / drop_rate
    # Get layer with desired sparsity
    sparse_enough = mean_sparsity <= factor
    sel_idx = np.argmax(sparse_enough * mean_sparsity)
    if sel_idx == 0 and np.sum(mean_sparsity) > 1: # sometimes first one is odd
        sparse_enough[0] = False
        sel_idx = np.argmax(sparse_enough * mean_sparsity)
    selected_weight = sparse_layer[sel_idx]
    selected_bias = biases[sel_idx]
    # only keep 5 per class on average
    weight_5_per_matrix = set_lowest_percent_to_zero(selected_weight,5)

    return weight_5_per_matrix,selected_bias


def set_lowest_percent_to_zero(matrix, keep_per):
    nonzero_indices = torch.nonzero(matrix)
    values = torch.tensor([matrix[x[0], x[1]] for x in nonzero_indices])
    sorted_indices = torch.argsort(torch.abs(values))
    total_allowed = int(matrix.shape[0] * keep_per)
    sorted_indices = sorted_indices[:-total_allowed]
    nonzero_indices_to_zero = [nonzero_indices[x] for x in sorted_indices]
    for to_zero in nonzero_indices_to_zero:
        matrix[to_zero[0], to_zero[1]] = 0
    return matrix