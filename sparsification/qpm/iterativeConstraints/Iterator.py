import torch
import numpy as np

class Iterator:
    def __init__(self, selected_features):
        self.total_relevant_features = selected_features
        self.prev_nonzeros = None
        self.prev_features = None
        self.last_selected_features = selected_features
        self.last_changed = 1

    def pre_optimize(self, edge_tensor, features, selected_features):
        self.prev_nonzeros = get_set_nonzeros(edge_tensor * features.X)
        self.prev_features = selected_features

    def get_relevant_features(self):
        return self.total_relevant_features

    def get_last_selection(self):
        return self.last_selected_features

    def changed_features(self):
        return  self.last_changed > 0

    def update(self, features, edge_tensor, ):
        if self.prev_nonzeros is None:
            raise ValueError("prev_nonzeros is None")
        selected_features = np.nonzero(np.isclose(features.X, np.ones_like(features.X)))[0]
        selected_set = set(selected_features)
        total_set = set(self.total_relevant_features)
        print("Number of totally new features ", len(selected_set - total_set))
        self.total_relevant_features = np.array(sorted(list(total_set.union(selected_set))))
        print("Number of total features relevant for deduplication", len(self.total_relevant_features))
        feature_diff = set(self.prev_features) - selected_set
        added_features = set()
        if len(feature_diff) == 0:
            print("No new features added")
        else:
            added_features = selected_set - set(self.prev_features)
            # for entry in added_features:
            print("Added features ", added_features)
            # for entry in feature_diff:
            print("Removed features ", feature_diff)

        nonzeros = get_set_nonzeros(edge_tensor * features.X)
        new_nonzeros = nonzeros - self.prev_nonzeros
        removed_entries = self.prev_nonzeros - nonzeros
        self.last_selected_features = selected_features
        new_entries = len(new_nonzeros)
        print("Changed Entries", new_entries)
        self.last_changed = new_entries
        print("Removed Connections to existing features",
              len([x for x in removed_entries if x[0] in selected_features]))
        print("Added Connections to old features", len([x for x in new_nonzeros if x[0] not in added_features]))



def get_set_nonzeros(tensor):
    nonzeros = set()
    total_nonzeros = torch.nonzero(tensor)
    for entry in total_nonzeros:
        nonzeros.add(tuple(entry.tolist()))
    return nonzeros
