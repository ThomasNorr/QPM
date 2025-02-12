import os
import pickle
from functools import partial

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB

from sparsification.qpm.clique_utils import find_minimum_viable_threshold, get_disallowed_vector_connections

from sparsification.qpm.iterativeConstraints.BalancedAssignment import ClassSparsity
from sparsification.qpm.iterativeConstraints.Iterator import Iterator
from sparsification.qpm.iterativeConstraints.deduplication import DeDuplipication


def solve_qp(similarity_measurement_matrix, cross_feature_similarity, feature_bias, target_features, features_per_class, bound=None, timelimit_in_hours_one_iter = 3, save_folder=None):
    rest = {}
    m = gp.Model("assignment")
    m.setParam('TimeLimit', timelimit_in_hours_one_iter * 60 * 60)
    if bound is not None:
        m.setParam('MIPGap', bound)
    if isinstance(similarity_measurement_matrix, torch.Tensor):
        similarity_measurement_matrix = similarity_measurement_matrix.cpu().numpy()
    if isinstance(cross_feature_similarity, torch.Tensor):
        cross_feature_similarity = cross_feature_similarity.cpu().numpy()

    edges = m.addMVar(similarity_measurement_matrix.shape, vtype=GRB.BINARY, name="edges")
    features = m.addMVar((similarity_measurement_matrix.shape[0], 1), vtype=GRB.BINARY, name="features")
    existing_edges = m.addMVar(similarity_measurement_matrix.shape, vtype=GRB.BINARY, name="existing_edges")
    m.addConstr(existing_edges == edges * features, "Restriction")
    m.addConstr(features.sum() == target_features, "LowDimensionality")
    n_classes = similarity_measurement_matrix.shape[1]
    # Initially only consider sparsity across all classes summed up
    m.addConstr((existing_edges).sum() <= n_classes * features_per_class, "Sparsity unbalanced")
    for class_idx in range(edges.shape[1]):
        m.addConstr((existing_edges[:, class_idx]).sum() >= 1, "Every Class has one feature")
    assignment_objective = (existing_edges * similarity_measurement_matrix).sum()
    main_objective = assignment_objective
    main_objective = main_objective / (features_per_class * n_classes) * 1000 # Scale to be in the same range as the other objective, no change for 5/50
    if cross_feature_similarity is not None:
        cross_feature_similarity[np.eye(cross_feature_similarity.shape[0], dtype=bool)] = 0
        save_path = save_folder /"cliques.pickle"
        if not os.path.exists(save_path):
            iter_corr, iters, predicted_size, init_clique, adj_matrix = find_minimum_viable_threshold(
                cross_feature_similarity,
                50)
            with open(save_path, "wb") as f:
                pickle.dump([iter_corr, iters, predicted_size, init_clique, adj_matrix], f)
        else:
            with open(save_path, "rb") as f:
                iter_corr, iters, predicted_size, init_clique, adj_matrix = pickle.load(f)
        rest["PercentSlack"] = iters
        rest["PredictedSize"] = predicted_size
        rest["LenMaxClique"] = len(init_clique)
        features_init = np.zeros(features.shape)
        features_init[list(init_clique)[:target_features]] = 1
        features.start = features_init
        rest["IterCorr"] = iter_corr
        all_disallowed = get_disallowed_vector_connections(cross_feature_similarity, iter_corr)
        additional_similarity = 0
        scaled_cross_feature_similarity = cross_feature_similarity - cross_feature_similarity[
            cross_feature_similarity != 0].min()
        scaled_cross_feature_similarity = scaled_cross_feature_similarity / scaled_cross_feature_similarity.max()
        for feature_1, features_2 in all_disallowed.items():
            assert scaled_cross_feature_similarity[feature_1, features_2].min() > 0
            additional_similarity += ((features[feature_1, 0] * features[features_2, 0]) *
                                      scaled_cross_feature_similarity[feature_1, features_2]).sum()
        main_objective = main_objective - additional_similarity
    if feature_bias is not None:
            main_objective += (features[:, 0] * feature_bias).sum()

    m.setObjective(main_objective, GRB.MAXIMIZE)
    m.optimize()
    initial_optimal_solution = m.objVal
    edge_tensor, selected_features, selected_edge_tensor = get_edge_features(edges,
                                                                             features)
    iterator = Iterator(selected_features)

    # Clean Iteration
    satisfied_iterators = []
    for this_iterator in [ ClassSparsity(iterator, m, features_per_class),  DeDuplipication(iterator, m, True)]:
        total_iterators = satisfied_iterators + [this_iterator]
        this_iterator.check_constraints(selected_edge_tensor, selected_features)
        final_run_done = False # Checks that the optimal solution is found, optimizes once more if the features changed at final iteration.
        while not final_run_done:
            while any([x.next_iter() for x in total_iterators]):
                final_run_done = False
                next_start = selected_edge_tensor.clone()
                for i, sngl_iterator in enumerate(total_iterators):
                    next_start = sngl_iterator.get_start_solution(next_start, similarity_measurement_matrix, edges,
                                                                  last_one=i == len(total_iterators) - 1)
                for sngl_iterator in total_iterators:
                    sngl_iterator.add_constraints(existing_edges, next_start, edges, features)
                edges.start = next_start
                features_start = np.zeros_like(features.X)
                features_start[selected_features] = 1
                features.start = features_start
                this_iterator.pre_optimize(edge_tensor, features, selected_features)
                m.optimize()
                edge_tensor, selected_features, selected_edge_tensor = get_edge_features(edges, features)
                this_iterator.after_optimization(features, edge_tensor)
                for sngl_iterator in total_iterators:
                    sngl_iterator.check_constraints(selected_edge_tensor, selected_features)
            if  this_iterator.same_features() or this_iterator.__class__.__name__ != "DeDuplipication":
                final_run_done = True

            else:
                print("Features did Change, rerun at Iterator", this_iterator.__class__.__name__)
                for sngl_iterator in total_iterators:
                    sngl_iterator.add_constraints(existing_edges, selected_edge_tensor, edges, features)
                this_iterator.pre_optimize(edge_tensor, features, selected_features)
                m.optimize()
                edge_tensor, selected_features, selected_edge_tensor = get_edge_features(edges, features)
                this_iterator.after_optimization(features, edge_tensor)
                for sngl_iterator in total_iterators:
                    sngl_iterator.check_constraints(selected_edge_tensor, selected_features)  #

            satisfied_iterators.append(this_iterator)
    print("Finished all Iterations")
    deduplicated_value = m.objVal
    rest["InitSolutionObj"] = initial_optimal_solution
    print("Objective Loss due to iterative optimization", initial_optimal_solution - deduplicated_value)
    nonzeros = len(torch.nonzero(selected_edge_tensor))
    print("Nonzeros ", nonzeros)
    assert nonzeros == features_per_class * n_classes
    assert len(selected_features) == target_features
    assert torch.min(torch.sum(selected_edge_tensor, dim=0)) >= features_per_class
    for iterator in satisfied_iterators:
        iterator.check_valid_tensor(selected_edge_tensor, selected_features)
    min_n_features_per_class = selected_edge_tensor.sum(dim=0).min()
    rest["GapOfSolution"] = m.MIPGAP
    rest["MainObjectiveValue"] = (selected_edge_tensor * similarity_measurement_matrix[selected_features]).sum()
    if feature_bias is not None:
        rest["LinearObjectiveValue"] = (feature_bias[selected_features]).sum()
    rest["TotalObjectiveValue"] = m.objVal
    print("Sparsest Class", min_n_features_per_class)
    min_n_class_per_fea = selected_edge_tensor.sum(dim=1).min()
    print("Sparsest Feature", min_n_class_per_fea)
    print("Log of QP", rest)
    return selected_features,selected_edge_tensor.T



def get_edge_features(edges, features):
    edge_array = edges.X
    edge_array = np.isclose(edge_array, np.ones_like(edge_array))
    edge_tensor = torch.tensor(edge_array, dtype=torch.bool)
    selected_features = np.nonzero(np.isclose(features.X, np.ones_like(features.X)))[0]
    selected_edge_tensor = edge_tensor[selected_features]
    return edge_tensor, selected_features, selected_edge_tensor
