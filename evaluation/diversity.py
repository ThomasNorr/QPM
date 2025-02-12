import numpy as np
import torch

from evaluation.helpers import softmax_feature_maps


class MultiKCrossChannelMaxPooledSum:
    def __init__(self, top_k_range, weights, interactions, func="softmax"):
        self.top_k_range = top_k_range
        self.weights = weights
        self.failed = False
        self.max_ks = self.get_max_ks(weights)
        self.locality_of_used_features = torch.zeros(len(top_k_range), device=weights.device)
        self.locality_of_exclusely_used_features = torch.zeros(len(top_k_range), device=weights.device)
        self.ns_k = torch.zeros(len(top_k_range), device=weights.device)
        self.exclusive_ns = torch.zeros(len(top_k_range), device=weights.device)
        self.interactions = interactions
        self.func = func

    def get_max_ks(self, weights):
        nonzeros = torch.count_nonzero(torch.tensor(weights), 1)
        return nonzeros

    def get_top_n_locality(self, outputs, initial_feature_maps, k):
        feature_maps, relevant_weights, vector_size, top_classes = self.adapt_feature_maps(outputs,
                                                                                           initial_feature_maps)
        max_ks = self.max_ks[top_classes]
        max_k_based_row_selection = max_ks >= k

        result = self.get_crosspooled(relevant_weights, max_k_based_row_selection, k, vector_size, feature_maps,
                                      separated=True)
        return result

    def get_locality(self, outputs, initial_feature_maps, n):
        answer = self.get_top_n_locality(outputs, initial_feature_maps, n)
        return answer

    def get_result(self):
        # if torch.sum(self.exclusive_ns) ==0:
        #     end_idx = len(self.exclusive_ns) - 1
        # else:

        exclusive_array = torch.zeros_like(self.locality_of_exclusely_used_features)
        local_array = torch.zeros_like(self.locality_of_used_features)
        # if self.failed:
        #     return local_array, exclusive_array
        cumulated = torch.cumsum(self.exclusive_ns, 0)
        end_idx = torch.argmax(cumulated)
        exclusivity_array = self.locality_of_exclusely_used_features[:end_idx + 1] / self.exclusive_ns[:end_idx + 1]
        exclusivity_array[exclusivity_array != exclusivity_array] = 0
        exclusive_array[:len(exclusivity_array)] = exclusivity_array
        locality_array = self.locality_of_used_features[self.locality_of_used_features != 0] / self.ns_k[
            self.locality_of_used_features != 0]
        local_array[:len(locality_array)] = locality_array
        return local_array, exclusive_array

    def get_crosspooled(self, relevant_weights, mask, k, vector_size, feature_maps, separated=False):
        relevant_indices = get_relevant_indices(relevant_weights, k)[mask]
        # this should have size batch x k x featuremapsize squared]
        indices = relevant_indices.unsqueeze(2).repeat(1, 1, vector_size)
        sub_feature_maps = torch.gather(feature_maps[mask], 1, indices)
        # shape batch x featuremapsquared: For each "pixel" the highest value
        cross_pooled = torch.max(sub_feature_maps, 1)[0]
        if separated:
            return torch.sum(cross_pooled, 1) / k
        else:
            ns = len(cross_pooled)
            result = torch.sum(cross_pooled) / (k)
            # should be batch x map size

            return ns, result

    def adapt_feature_maps(self, outputs, initial_feature_maps):
        if self.func == "softmax":
            feature_maps = softmax_feature_maps(initial_feature_maps)
        elif self.func == "SumNMax":
            feature_maps = initial_feature_maps / torch.clamp(torch.mean(torch.abs(initial_feature_maps),
                                                                         dim=(2, 3), keepdim=True), min=1e-5)
            feature_maps = softmax_feature_maps(feature_maps)
        feature_maps = torch.flatten(feature_maps, 2)
        vector_size = feature_maps.shape[2]
        top_classes = torch.argmax(outputs, dim=1)
        relevant_weights = self.weights[top_classes]
        if relevant_weights.shape[1] != feature_maps.shape[1]:
            feature_maps = self.interactions.get_localized_features(initial_feature_maps)
            feature_maps = softmax_feature_maps(feature_maps)
            feature_maps = torch.flatten(feature_maps, 2)
        return feature_maps, relevant_weights, vector_size, top_classes

    def calculate_locality(self, outputs, initial_feature_maps):
        feature_maps, relevant_weights, vector_size, top_classes = self.adapt_feature_maps(outputs,
                                                                                           initial_feature_maps)
        max_ks = self.max_ks[top_classes]
        for k in self.top_k_range:
            # relevant_k_s = max_ks[]
            max_k_based_row_selection = max_ks >= k
            if torch.sum(max_k_based_row_selection) == 0:
                break

            exclusive_k = max_ks == k
            if torch.sum(exclusive_k) != 0:
                ns, result = self.get_crosspooled(relevant_weights, exclusive_k, k, vector_size, feature_maps)
                self.locality_of_exclusely_used_features[k - 1] += result
                self.exclusive_ns[k - 1] += ns
            ns, result = self.get_crosspooled(relevant_weights, max_k_based_row_selection, k, vector_size, feature_maps)
            self.ns_k[k - 1] += ns
            self.locality_of_used_features[k - 1] += result

    def __call__(self, outputs, initial_feature_maps):
        self.calculate_locality(outputs, initial_feature_maps)


def get_relevant_indices(weights, top_k):
    top_k = weights.topk(top_k)[1]
    return top_k