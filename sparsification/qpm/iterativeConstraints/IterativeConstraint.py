import numpy as np
import torch


def get_start_solution(changed_tensor, edges, selected_features):
    initital_start_edges = np.zeros_like(edges.X)
    initital_start_edges[selected_features] = changed_tensor
    return initital_start_edges

class IterativeConstraint:
    def __init__(self, iterator, model, parameter):
        self.iterator = iterator
        self.model = model
        self.parameter = parameter
        self.prev_constraints = []
        # self.satisfied_iterators = satifised_iterators

    def pre_optimize(self, edge_tensor, features, selected_features):
        self.iterator.pre_optimize(edge_tensor, features, selected_features)

    def add_constraints(self, existing_edges, next_start, edges, features):
        pass

    def check_constraints(self, selected_edge_tensor, selected_features):
        pass

    def compute_start_solution(self, selected_edge_tensor, similarity_measurement_matrix):
        pass

    def same_features(self):
        return not self.iterator.changed_features()

    def check_valid_tensor(self, selected_edge_tensor, selected_features):
        print("Checking valid tensor for ", self.__class__.__name__)
        self.check_constraints(selected_edge_tensor, selected_features)
        assert not self.next_iter(), "Invalid tensor for " + self.__class__.__name__
        print("Valid tensor for ", self.__class__.__name__)

    def get_start_solution(self, selected_edge_tensor, similarity_measurement_matrix, edges, last_one):
        print("Computing start Solution for Iterative Constraint ", self.__class__.__name__)
        if self.next_iter():
            unsparsified_edge_tensor = self.compute_start_solution(selected_edge_tensor, similarity_measurement_matrix[
                self.iterator.get_last_selection()])
        else:
            unsparsified_edge_tensor = selected_edge_tensor
        initial_start_edges = unsparsified_edge_tensor
        #   self.check_valid_tensor(torch.tensor(initial_start_edges))
        if last_one:
            initial_start_edges = get_start_solution(unsparsified_edge_tensor, edges,
                                                     self.iterator.get_last_selection())

        print("Finished Computing start Solution for Iterative Constraint ", self.__class__.__name__)
        return initial_start_edges

    def after_optimization(self, features, edge_tensor):
        self.iterator.update(features, edge_tensor)

    def next_iter(self):
        raise NotImplementedError("next_iter not implemented")
