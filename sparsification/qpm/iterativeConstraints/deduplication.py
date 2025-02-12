
from collections import defaultdict
import time
import gurobipy as gp
import numpy as np
import sklearn.metrics
import torch
from gurobipy import GRB


from sparsification.qpm.iterativeConstraints.IterativeConstraint import IterativeConstraint


class DeDuplipication(IterativeConstraint):
    def __init__(self, iterator, model, parameter):
        super().__init__(iterator, model, parameter)
        self.prev_dubs = []

    def add_constraints(self, existing_edges, next_start, edges, features):
        diffs, us, self.prev_constraints, self.prev_dubs = clean_add_uniqueness_constraint(self.model, self.duplicates,
                                                                                           self.prev_constraints,
                                                                                           self.iterator.get_relevant_features(),
                                                                                           self.prev_dubs,
                                                                                           existing_edges,
                                                                                           next_start)

    def check_constraints(self, selected_edge_tensor, selected_features):
        self.duplicates, self.relevant_classes = get_duplicates(selected_edge_tensor)

    def compute_start_solution(self, selected_edge_tensor, selected_similarity_measurement_matrix):
        print("Deduplicating without maintaing feature sparsity")
        return sophisticated_deduplication(torch.tensor(selected_edge_tensor),
                                           torch.tensor(selected_similarity_measurement_matrix))

    def next_iter(self):
        print("Starting to remove", len(self.duplicates), "Current Duplicates from ", len(self.relevant_classes),
              "Classes")
        return len(self.relevant_classes) > 0





def get_duplicates(selected_edge_tensor):
    duplicates = []
    relevant_classes = set()
    for class_idx in range(selected_edge_tensor.shape[1]):
        for second_idx in range(class_idx + 1, selected_edge_tensor.shape[1]):
            if (selected_edge_tensor[:, class_idx] == selected_edge_tensor[:, second_idx]).all():
                duplicates.append((class_idx, second_idx))
                relevant_classes.add(class_idx)
                relevant_classes.add(second_idx)
    return duplicates, relevant_classes


class CheckDuplicates:
    def __init__(self, selected_edge_tensor):
        self.selected_edge_tensor = selected_edge_tensor
        self.nonzeros = (selected_edge_tensor != 0).sum(dim=0)
        self.nonzero_to_index_dict = {i: np.nonzero(self.nonzeros == i) for i in np.unique(self.nonzeros)}

    def would_line_be_duplicate(self, questionable_class_index, questionable_feature_index):
        questionable_line = self.selected_edge_tensor[:, questionable_class_index].clone()
        questionable_line[questionable_feature_index] = 1
        entries = len(questionable_line.nonzero())
        if not entries in self.nonzero_to_index_dict:
            return False
        relevant_classes = self.nonzero_to_index_dict[entries]
        for second_idx in relevant_classes:
            if (questionable_line == self.selected_edge_tensor[:, second_idx]).all():
                return True
        return False


def would_line_be_duplicate(selected_edge_tensor, questionable_class_index, questionable_feature_index):
    questionable_line = selected_edge_tensor[:, questionable_class_index]
    questionable_line[questionable_feature_index] = 1
    for second_idx in range(selected_edge_tensor.shape[1]):
        if second_idx == questionable_class_index:
            continue
        if (questionable_line == selected_edge_tensor[:, second_idx]).all():
            return True
    return False


def check_duplicate(selected_edge_tensor, similarity_matrix, raise_error=False, raise_warning=False,
                    second_order_min=False, no_duplicates=True):
    duplicates, relevant_classes = get_duplicates(selected_edge_tensor)
    if len(duplicates) == 0:
        print("No duplicates")
    else:
        print("Duplicates", duplicates)
        if raise_error:
            raise ValueError("Duplicates")
        elif raise_warning:
            print("WARNING: Duplicates")
        else:
            similarity_matrix = torch.tensor(similarity_matrix)
            remining_duplicates = duplicates
            pre_deduplication = selected_edge_tensor.clone()
            added = set()
            idx = 0
            if second_order_min:
                print("Directly trying sophisticated deduplication")
                return sophisticated_deduplication(pre_deduplication, similarity_matrix)
            min_number_of_entries = second_order_min if second_order_min else 1
            while len(remining_duplicates) > 0:
                if no_duplicates:
                    noDuplicate_Checker = CheckDuplicates(selected_edge_tensor)
                idx = idx % len(remining_duplicates)
                c1, c2 = remining_duplicates[idx]
                # if selected_edge_tensor[:, c1] == selected_edge_tensor[:, c2]:
                #     idx += 1
                #     idx = idx % len(remining_duplicates)
                #     continue
                assigned_edges = selected_edge_tensor * similarity_matrix
                nonzeros_assigned = (assigned_edges != 0).sum(dim=0)
                possible_remover_classes = torch.tensor(sorted(list(
                    set(torch.nonzero(nonzeros_assigned > min_number_of_entries)[:, 0].tolist()) - set([c1, c2]).union(
                        added))))
                if len(possible_remover_classes) == 0:
                    # print("No possible remover classes, returning not deduplicated")
                    # return  pre_deduplication
                    print("No possible remover classes, trying sophisticated deduplication")
                    return sophisticated_deduplication(pre_deduplication, similarity_matrix)
                possible_removed_edges = assigned_edges[:, possible_remover_classes]
                max_sim = torch.min(possible_removed_edges)  # initialise to lowest
                possible_removed_edges[possible_removed_edges == 0] = torch.max(possible_removed_edges) + 1

                for sngl_class in [c1, c2]:
                    relevant_row = similarity_matrix[:, sngl_class] * (~selected_edge_tensor[:, sngl_class])
                    if no_duplicates:
                        would_be_duplicate = True
                        possible_features = np.arange(relevant_row.shape[0])
                        while would_be_duplicate:
                            best_new, best_indices = torch.max(relevant_row[possible_features], dim=0)
                            this_feature_idx = possible_features[best_indices]
                            possible_features = np.delete(possible_features, best_indices)
                            would_be_duplicate = noDuplicate_Checker.would_line_be_duplicate(sngl_class,
                                                                                             this_feature_idx)
                            if best_new >= max_sim and not would_be_duplicate:
                                max_sim = best_new
                                best_class = sngl_class
                                best_idx = this_feature_idx
                    else:
                        best_new, best_indices = torch.max(relevant_row, dim=0)
                        if best_new >= max_sim:
                            max_sim = best_new
                            best_class = sngl_class
                            best_idx = best_indices
                # New entry found at best_idx; best_class
                row, col = np.unravel_index(torch.argmin(possible_removed_edges), possible_removed_edges.shape)
                selected_edge_tensor[row, possible_remover_classes[col]] = False
                selected_edge_tensor[best_idx, best_class] = True
                added.add(best_class)
                remining_duplicates, this_relevant_classes = get_duplicates(selected_edge_tensor)
                # print("Removed Duplicate ", (c1, c2), "by replacing", (row, possible_remover_classes[col]),
                #       similarity_matrix[row, possible_remover_classes[col]], "with", (best_idx, best_class),
                #       similarity_matrix[best_idx, best_class], "Remaining ", len(remining_duplicates), "duplicates ",
                #       remining_duplicates)
                idx += 1

            print("Duplicates removed Manually")
            check_duplicate(selected_edge_tensor, similarity_matrix, raise_error=True)

    return selected_edge_tensor


class SwapDuplicateChecker:
    def __init__(self, selected_edge_tensor):
        self.selected_edge_tensor = selected_edge_tensor
        self.distance_matrix = sklearn.metrics.pairwise_distances(selected_edge_tensor.T, metric='l1')
        self.distance_matrix[np.eye(self.distance_matrix.shape[0], dtype=bool)] = np.inf

    def would_line_be_duplicate(self, questionable_class_index, questionable_feature_index, remove_from_class_index):
        questionable_line = self.selected_edge_tensor[:, questionable_class_index].clone()
        questionable_line[questionable_feature_index] = 1
        questionable_line[remove_from_class_index] = 0
        relevant_classes = np.nonzero(self.distance_matrix[questionable_class_index] == 2)[0]
        for second_idx in relevant_classes:
            if (questionable_line == self.selected_edge_tensor[:, second_idx]).all():
                return True
        return False


def sophisticated_deduplication(selected_edge_tensor, similarity_matrix):
    # GOALS: Remove duplicates by:  Solve duplicate
    # by removing one at a time via changing lowest assigned to highest unassigned
    # if unassigned does not introduce new duplicates. Since we want to maintain min sparsity, we do not change sparsity.
    print("Sophisticated Deduplication")
    remining_duplicates, relevant_classes = get_duplicates(selected_edge_tensor)
    print("Remaining Duplicates Before Sophisticated Deduplication ", len(remining_duplicates))
    start_time = time.time()
    idx = 0
    total_cost = 0
    while len(remining_duplicates) > 0:
        noDuplicate_Checker = SwapDuplicateChecker(selected_edge_tensor)
        if idx >= len(remining_duplicates):
            raise ValueError("Sophisticated Deduplication failed")
        c1, c2 = remining_duplicates[idx]
        swap_found = False
        min_cost = torch.max(similarity_matrix) + 1
        for sngl_class in [c1, c2]:
            this_class_tensor = selected_edge_tensor[:, sngl_class] * similarity_matrix[:, sngl_class]
            relevant_row = similarity_matrix[:, sngl_class] * (~selected_edge_tensor[:, sngl_class])
            assigned_features = torch.nonzero(this_class_tensor).squeeze(-1)
            # if len(assigned_features.shape) > 1:
            #     assigned_features = assigned_features
            # else:
            #     print("sdas")
            for remove_feature in assigned_features:
                would_be_duplicate = True
                assignable_features = torch.nonzero(~selected_edge_tensor[:, sngl_class]).squeeze()

                while would_be_duplicate:
                    best_new, best_indices = torch.max(relevant_row[assignable_features], dim=0)
                    this_feature_idx = assignable_features[best_indices]
                    assignable_features = np.delete(assignable_features, best_indices)
                    would_be_duplicate = noDuplicate_Checker.would_line_be_duplicate(sngl_class, this_feature_idx,
                                                                                     remove_feature)
                    if not would_be_duplicate:
                        total_cost = this_class_tensor[remove_feature] - relevant_row[this_feature_idx]
                        if total_cost <= min_cost:
                            swap_found = True
                            min_cost = total_cost
                            best_class = sngl_class
                            best_idx = this_feature_idx
                            to_remove = remove_feature
        idx += 1
        if not swap_found:
            print("No feasible swap found for this pair of duplicates")
            continue
        selected_edge_tensor[to_remove, best_class] = False
        selected_edge_tensor[best_idx, best_class] = True
        idx = 0
        total_cost += min_cost
        remining_duplicates, relevant_classes = get_duplicates(selected_edge_tensor)
        print("Removed Duplicate ", (c1, c2), "by replacing", (to_remove, best_class),
              similarity_matrix[to_remove, best_class], "with", (best_idx, best_class),
              similarity_matrix[best_idx, best_class], "Remaining ", len(remining_duplicates), "duplicates ",
              remining_duplicates)

    duplicates, relevant_classes = get_duplicates(selected_edge_tensor)
    print("Total time taken : ", time.time() - start_time, "seconds")
    print("Remaining Duplicates after Sophisticated Deduplication ", len(duplicates))
    print("Cost of sophisticated Deduplication ", total_cost)
    return selected_edge_tensor


def clean_add_uniqueness_constraint(m, duplicates, previous_constraints, total_relevant,
                                    prev_dubs, existing_edges, existing_edge_tensor,
                                    ):
    diffs = []
    us = []
    full_duplicates = {(c1, c2): prev_dubs[(c1, c2)] for c1, c2 in prev_dubs}
    for c1, c2 in duplicates:
        equal_entries1 = np.nonzero(existing_edge_tensor[:, c1])
        equal_entries2 = np.nonzero(existing_edge_tensor[:, c2])
        equal_entries = np.union1d(equal_entries1, equal_entries2)
        if (c1, c2) in prev_dubs:
            equal_entries = np.union1d(equal_entries, prev_dubs[(c1, c2)])
        full_duplicates[(c1, c2)] = equal_entries
    duplicates = set(duplicates).union(
        set(prev_dubs))  # prev_dubs needs to be a dict with (c1,c2): features previsouly duplicated

    duplicates = list(duplicates)
    constraints = []
    for constraint in previous_constraints:
        m.remove(constraint)

    feature_counter = DuplicateCounter(full_duplicates)
    for class_idx, second_class in duplicates:
        feature_indices = full_duplicates[(class_idx, second_class)]
        if feature_counter.is_infeasible(feature_indices):
            feature_indices = total_relevant
        diff = m.addMVar(len(feature_indices), lb=-1, vtype=GRB.INTEGER, name="diff")
        u = m.addVar(name="u", vtype=GRB.INTEGER)
        constraints.append(m.addConstr(diff == existing_edges[feature_indices, class_idx] - existing_edges[
            feature_indices, second_class], f"Aux_{class_idx}_{second_class}"))
        constraints.append(m.addConstr(u == gp.norm(diff, 1), f"Norm_Deduplication_{class_idx}_{second_class}"))
        constraints.append(m.addConstr(u >= 1, f"Deduplication_{class_idx}_{second_class}"))
        us.append(u)
        diffs.append(diff)

    return diffs, us, constraints, full_duplicates

class DuplicateCounter:
    base = 2

    def __init__(self, duplicate_dict):
        self.duplicates = defaultdict(lambda: 0)
        self.result_dict = self.calculate_feasiblity(duplicate_dict)

    def add_features(self, features):
        differentiator = self.converet_array_to_key(features)
        self.duplicates[differentiator] += 1

    def converet_array_to_key(self, array):
        return tuple(sorted(array))

    def calculate_feasiblity(self, duplicate_dict):
        for key, features in duplicate_dict.items():
            self.add_features(features)
        answer_dict = {}
        for key, counter in self.duplicates.items():
            answer = False
            if counter > self.base ** len(key):
                answer = True
            answer_dict[key] = answer
        return answer_dict

    def is_infeasible(self, features):
        key = self.converet_array_to_key(features)
        if key not in self.result_dict:
            return False
        return self.result_dict[self.converet_array_to_key(features)]


if __name__ == '__main__':
    n_classes = 10
    n_features = 50
    test_duplicate_array = np.random.randint(0, 2, (n_features, n_classes))
    test_duplicate_array[:, 3:7] = test_duplicate_array[:, 0]
    sophisticated_deduplication(test_duplicate_array)
